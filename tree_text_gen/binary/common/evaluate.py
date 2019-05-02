from tqdm import tqdm
from collections import defaultdict
import torch
import tree_text_gen.binary.common.data as data
import tree_text_gen.binary.common.tree as tree_util


def convert_samples(samples, ground_truth, i2tok, tok2i):
    if not isinstance(samples, list):
        samples = samples.clone().cpu().tolist()
    if ground_truth is not None:
        ground_truth = ground_truth.clone().cpu().tolist()
    converted = []
    for i in range(len(samples)):
        sample = samples[i]
        raw_tokens = data.inds2toks(i2tok, sample)
        root = tree_util.build_tree(raw_tokens)
        inorder_tokens, nodes = tree_util.tree_to_text(root)
        tokens_levels = [(node.value, node.level) for node in nodes]
        output = {'raw_tokens': raw_tokens,
                  'tree_string': tree_util.print_tree(root),
                  'inorder_tokens': inorder_tokens,
                  'genorder_tokens': get_genorder_tokens(raw_tokens),
                  'token_levels': tokens_levels}
        if ground_truth is not None:
            gt_inds = [x for x in ground_truth[i] if x != tok2i['</s>'] and x != tok2i['<p>']]
            gt_tokens = data.inds2toks(i2tok, gt_inds)
            output['gt_tokens'] = gt_tokens

        converted.append(output)
    return converted


def get_genorder_tokens(raw_tokens):
    # Generation ended when the number of end labels exceeds the number of labels
    nl = 0
    ne = 0
    end = 0
    for i, t in enumerate(raw_tokens):
        if t != '<end>':
            nl += 1
        else:
            ne += 1
        if ne > nl:
            end = i
            break
    genorder = [t for t in raw_tokens[:end] if t != '<end>']
    return genorder


def predict_batch(model, batch):
    with torch.no_grad():
        model.eval()
        if model.model_type == 'transformer':
            scores, samples = model(xs=batch.src, ys=None, p_oracle=None, num_samples=len(batch))
        else:
            if model.model_type == 'bagorder':
                xs, annots = batch
                xs = xs.to(model.device)
                num_samples = None
            elif model.model_type == 'translation':
                xs = batch.src
                num_samples = xs[0].size(0)
            scores, samples = model.forward(xs, oracle=None, num_samples=num_samples)

        model.train()
    return scores, samples


def save_tokens(list_of_tokenlists, filepath):
    with open(filepath, "a", encoding="utf-8") as f:
        for ts in list_of_tokenlists:
            f.write(' '.join(ts))
            f.write('\n')


def eval_sacrebleu(hyp='__hyp.txt', ref='__ref.txt', remove_files=True):
    from subprocess import Popen, PIPE
    from shlex import split
    import re
    import os
    p1 = Popen(split("cat %s" % hyp), stdout=PIPE)
    p2 = Popen(split("sacrebleu --smooth none %s" % ref), stdin=p1.stdout, stdout=PIPE)
    out, err = p2.communicate()
    out = out.decode('utf-8')
    print(out)
    bleu = float(re.findall('version\.1\.2\.12\s=\s([\d]+\.[0-9]+)', out)[0])
    if remove_files:
        os.remove(hyp)
        os.remove(ref)
    return bleu


def eval_ribes(hyp='__hyp.txt', ref='__ref.txt', override_options=dict()):
    import tree_text_gen.binary.translation.evaluation.ribes as ribes
    from tree_text_gen.binary.common.util import dotdict
    import io
    # Default options from ribes.py, plus the given ref:
    options = dotdict({
        'case': False,
        'sentence': False,
        'alpha': 0.25,
        'beta': 0.10,
        'emptyref': False,
        'ref': [ref]
    })
    for k, v in override_options.items():
        options[k] = v
    args = [hyp]
    file = io.StringIO()
    ribes.outputRIBES(options, args, file=file)
    out = file.getvalue()
    file.close()
    metric = float(out.split(' ')[0])
    return metric


def eval_meteor(hyp='__hyp.txt', ref='__ref.txt', meteor_jar='/home/sw1986/libraries/meteor-1.5/meteor-1.5.jar',
                target_language='en'):
    """Download the jar from: http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz"""
    from subprocess import Popen, PIPE
    from shlex import split
    cmd = 'java -Xmx2G -jar %s "%s" "%s" -l %s -norm' % (meteor_jar, hyp, ref, target_language)
    p2 = Popen(split(cmd), stdout=PIPE)
    out, err = p2.communicate()
    out = out.decode('utf-8')
    metric = float(out.strip().split('\n')[-1].split(' ')[-1])
    return metric


def eval_yisi(w2v_path='/home/sw1986/datasets/word2vec/yisi/en/unit.d300.en.bin', hyp='__hyp.txt', ref='__ref.txt', override_options=dict()):
    """Assumes YISI_HOME path has been set. See: https://github.com/chikiulo/yisi.
    Download a relevant word2vec file from http://chikiu-jackie-lo.org/home/index.php/yisi"""
    from subprocess import Popen, PIPE
    from shlex import split
    import os
    options = {
        'srclang': 'de',
        'tgtlang': 'en',
        'lexsim-type': 'w2v',
        'outlexsim-path': w2v_path,
        'reflexweight-type': 'learn',
        'phrasesim-type': 'nwpr',
        'ngram-size': '3',
        'mode': 'yisi',
        'alpha': '0.8',
        'ref-file': ref,
        'hyp-file': hyp,
        'sntscore-file': 'out.sntyisi1',
        'docscore-file': 'out.docyisi1',
    }
    for k, v in override_options.items():
        options[k] = v
    yisi_config = '\n'.join(['='.join([k, v]) for k, v in options.items()])
    with open('__yisi_config_temp.config', 'w') as f:
        f.write(yisi_config)

    yisi = os.path.join(os.environ['YISI_HOME'], 'bin', 'yisi')
    cmd = '%s --config __yisi_config_temp.config' % yisi
    p2 = Popen(split(cmd), stdout=PIPE)
    out, err = p2.communicate()
    out = out.decode('utf-8')
    metric = float(open('out.docyisi1', 'r').read().strip())

    os.remove(options['sntscore-file'])
    os.remove(options['docscore-file'])
    os.remove('__yisi_config_temp.config')
    return metric


def eval_by_length(model, dataloader):
    model.eval()
    l_to_preds = defaultdict(list)
    l_to_gt = defaultdict(list)
    for i, batch in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        scores, samples = predict_batch(model, batch)
        if model.model_type == 'translation':
            gt = batch.trg[0]
        elif model.model_type == 'bagorder':
            gt = batch[0]
        outs = convert_samples(samples, gt, model.i2tok, model.tok2i)
        for out in outs:
            gt = out['gt_tokens']
            l = len(gt)
            l_to_preds[l].append(out['inorder_tokens'])
            l_to_gt[l].append(gt)

    results = {}
    for l in sorted(list(l_to_preds.keys())):
        save_tokens(l_to_preds[l], '__hyp.txt')
        save_tokens(l_to_preds[l], '__ref.txt')
        b = eval_sacrebleu('__hyp.txt', '__ref.txt')
        results[l] = b
        print(l, b)
    return results