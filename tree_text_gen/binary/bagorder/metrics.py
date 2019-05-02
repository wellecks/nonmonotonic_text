# --- metrics & evaluation
from collections import defaultdict, Counter

import numpy as np
from sacrebleu import corpus_bleu

from tree_text_gen.binary.common.data import inds2toks
from tree_text_gen.binary.common.tree import build_tree, tree_to_text
import os


class Metrics(object):
    def __init__(self, tok2i, i2tok, bleu_to_file=True):
        self._metrics = defaultdict(float)
        self._tok2i = tok2i
        self._i2tok = i2tok
        self.bleu_to_file = bleu_to_file
        if self.bleu_to_file:
            if os.path.exists('__hyp.txt'):
                os.remove('__hyp.txt')
            if os.path.exists('__ref.txt'):
                os.remove('__ref.txt')
            self._hyp_txt = open('__hyp.txt', 'a', encoding='utf-8')
            self._ref_txt = open('__ref.txt', 'a', encoding='utf-8')

    def reset(self):
        self._metrics = defaultdict(float)

    def update(self, scores, samples, batch):
        for i in range(batch[0].size(0)):
            tokens = inds2toks(self._i2tok, samples[i].cpu().tolist())
            root = build_tree(tokens)
            tokens, nodes = tree_to_text(root)
            gt_inds = [x for x in batch[0][i].cpu().tolist() if x != self._tok2i['</s>'] and x != self._tok2i['<p>']]
            gt_tokens = inds2toks(self._i2tok, gt_inds)

            self._metrics['em'] += self._exact_match(tokens, gt_tokens)
            precision, recall, f1 = self._prec_recall_f1_score(tokens, gt_tokens)

            # BLEU
            bleu_results = self._sentence_bleu(tokens, gt_tokens)
            self._metrics['bleu'] += bleu_results.score
            self._metrics['precisions-1'] += bleu_results.precisions[0]
            self._metrics['precisions-2'] += bleu_results.precisions[1]
            self._metrics['precisions-3'] += bleu_results.precisions[2]
            self._metrics['precisions-4'] += bleu_results.precisions[3]
            self._metrics['brevity_penalty'] += bleu_results.bp
            self._metrics['sys_len'] += bleu_results.sys_len
            self._metrics['ref_len'] += bleu_results.ref_len

            # Sentence-level averages scores over sentences
            self._metrics['precision'] += precision
            self._metrics['recall'] += recall
            self._metrics['f1'] += f1

            self._metrics['depth_score'] += self._depth_score(nodes)
            self._metrics['avg_span'] += self._avg_span(nodes)

            # Normalizer for the above summed metrics.
            self._metrics['n_sent'] += 1.0

            if self.bleu_to_file:
                self._save_tokens(tokens, gt_tokens)


        self._metrics['n_batch'] += 1.0

    def _save_tokens(self, tokens, gt_tokens):
        hyp = np.array(tokens, dtype=str)
        ref = np.array(gt_tokens, dtype=str)
        if len(hyp)>0 and len(ref)>0:
            np.savetxt(self._hyp_txt, [np.array(tokens, dtype=str)], delimiter=" ", fmt="%s", encoding='utf-8')
            np.savetxt(self._ref_txt, [np.array(gt_tokens, dtype=str)], delimiter=" ", fmt="%s", encoding='utf-8')

    def _depth_score(self, nodes):
        # 0 => balanced (depth is logn)
        # 1 => chain (i.e. no node has 2 children; note this is not necessarily left-to-right or right-to-left)
        levels = [node.level for node in nodes]
        max_depth = max(levels)+1  # +1 is for 0-indexing
        n = len(nodes)
        score = (max_depth - np.log2(n)) / (n - np.log2(n))
        return score

    @staticmethod
    def _avg_span(nodes):
        # Average number of children (non <end> nodes) for non-leaf nodes.
        n_children = []
        for node in nodes:
            nc = int(node.left is not None) + int(node.right is not None)
            if nc != 0:  # not a leaf node
                n_children.append(nc)
        avg_span = np.sum(n_children) / float(max(1, len(n_children)))
        return avg_span

    @staticmethod
    def _exact_match(pred_tokens, gt_tokens):
        return float(tuple(pred_tokens) == tuple(gt_tokens))

    @staticmethod
    def _prec_recall_f1_score(pred_tokens, gt_tokens):
        # ref: https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/metrics.py
        common = Counter(gt_tokens) & Counter(pred_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def _sentence_bleu(pred_tokens, gt_tokens):
        hypothesis = ' '.join(pred_tokens)
        reference = ' '.join(gt_tokens)
        bleu = corpus_bleu(hypothesis, reference,
                             smooth='none',
                             use_effective_order=True)
        return bleu

    def report(self, kind='train', round_level=4):
        if self.bleu_to_file:
            self._hyp_txt.close()
            self._ref_txt.close()
        metrics = {}
        for m in ['f1', 'precision', 'recall', 'em', 'depth_score', 'bleu', 'avg_span',
                  'brevity_penalty', 'sys_len', 'ref_len',
                  'precisions-1', 'precisions-2', 'precisions-3', 'precisions-4']:
            metrics['%s' % m] = self._metrics['%s' % m] / max(1.0, float(self._metrics['n_sent']))

        metrics_ = {}
        for m in metrics:
            metrics_['%s/%s' % (kind, m)] = metrics[m]
        metrics = metrics_
        for m in metrics:
            metrics[m] = round(metrics[m], round_level)
        return metrics

    def log(self, metrics, kind, metric_names):
        s = ''
        for name in metric_names:
            s += '%s/%s %.3f\t' % (kind, name, metrics['%s/%s' % (kind, name)])
        return s


