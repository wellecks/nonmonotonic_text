import json
import torch
import os
import pickle
import numpy as np
import torch as th
import tree_text_gen.binary.common.samplers as samplers
from pprint import pprint as pp
from tqdm import tqdm
from itertools import product
from tree_text_gen.binary.common.model import LSTMDecoder
from tree_text_gen.binary.common.encoder import LSTMEncoder
import tree_text_gen.binary.common.util as util
import tree_text_gen.binary.common.tree as tree_util
import tree_text_gen.binary.common.evaluate as common_eval
from tree_text_gen.binary.translation.metrics import Metrics


def load_model_eval(expr_dir, expr_name, checkpoint=False):
    model_file = os.path.join(expr_dir, expr_name + ('.checkpoint' if checkpoint else ''))
    tok2i = json.load(open(os.path.join(expr_dir, 'tok2i.json'), 'r'))
    config = json.load(open(os.path.join(expr_dir, 'model_config.json'), 'r'))
    config['device'] = th.device('cpu') if config['device'] == 'cpu' or not th.cuda.is_available() else th.device('cuda:0')
    config['model_type'] = config.get('model_type', 'translation')
    encoder = LSTMEncoder(config, tok2i)
    sampler = samplers.GreedySampler()
    rollin_sampler = samplers.MixedRollin(0.0, sampler, sampler, 'trajectory')
    model = LSTMDecoder(config, tok2i, rollin_sampler, encoder).to(config['device'])
    if config['device'] == 'cpu' or not th.cuda.is_available():
        state_dict = th.load(model_file, map_location='cpu')
    else:
        state_dict = th.load(model_file)
    if checkpoint:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    i2tok = {j: i for i, j in tok2i.items()}
    model.tok2i = tok2i
    model.i2tok = i2tok
    return model


def load_transformer_eval(expr_dir, expr_name, checkpoint=False, parallel=True):
    import tree_text_gen.binary.common.transformer as transformer
    tok2i = json.load(open(os.path.join(expr_dir, 'tok2i.json'), 'r'))
    i2tok = {j: i for i, j in tok2i.items()}

    # backwards compatibility for previous json format (we only device and longest_label from it)
    if os.path.exists(os.path.join(expr_dir, 'model_config.json')):
        model_config = json.load(open(os.path.join(expr_dir, 'model_config.json'), 'r'))
        model_config['device'] = th.device('cpu') if model_config['device'] == 'cpu' or not th.cuda.is_available() else th.device('cuda:0')

        # hard-coded!!
        if 'uniform' in expr_dir:
            parallel = False

        config = transformer.SmallConfig(model_config['device'])
        config.longest_label = model_config['longest_label']

        # hard-coded!!
        if 'annealed_tree' in expr_dir:
            config.tree_encoding = True
        else:
            config.tree_encoding = False
        config.auxiliary_end = True
    else:
        config = pickle.load(open(os.path.join(expr_dir, 'model_config.pkl'), 'rb'))

    model = transformer.Transformer(config, tok2i, tok2i, loss_fn=None)
    model.tok2i = tok2i
    model.i2tok = i2tok
    model.model_type = 'transformer'

    if parallel:
        model = util.DataParallel(model, device_ids=[0])
    model.cuda()

    model_file = os.path.join(expr_dir, expr_name + ('.checkpoint' if checkpoint else ''))
    state_dict = torch.load(model_file)
    if checkpoint:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    return model


def eval_single(model, sentence, TRG=None):
    scores, preds = model.forward(xs=sentence.src, oracle=None, num_samples=1)
    raw_tokens = [x.split() for x in TRG.reverse(preds, unbpe=True)][0]
    model.train()

    root = tree_util.build_tree(raw_tokens)
    inorder_tokens, nodes = tree_util.tree_to_text(root)
    tokens_levels = [(node.value, node.level) for node in nodes]

    if TRG is not None:
        trg = [x.split() for x in TRG.reverse(sentence.trg[0], unbpe=True)][0]
        src = [x.split() for x in TRG.reverse(sentence.src[0], unbpe=True)][0]
    else:
        gt_inds = [x for x in sentence.trg[0][0].cpu().tolist() if x != model.tok2i['</s>']
                   and x != model.tok2i['<p>']
                   and x != model.tok2i['<s>']]
        trg = TRG.inds2toks(model.i2tok, gt_inds)

        src_inds = [x for x in sentence.src[0][0].cpu().tolist() if x != model.tok2i['</s>']
                    and x != model.tok2i['<p>']
                    and x != model.tok2i['<s>']]
        src = TRG.inds2toks(model.i2tok, src_inds)

    output = {'raw_tokens': raw_tokens,
              'tree_string': tree_util.print_tree(root),
              'inorder_tokens': inorder_tokens,
              'genorder_tokens': common_eval.get_genorder_tokens(raw_tokens),
              'token_levels': tokens_levels,
              'gt_tokens': trg,
              'src_tokens': src}
    return output, scores, preds


def eval_dataset(model, dataloader, bleu_to_file=True, log=True, **kwargs):
    model.eval()
    vmetrics = Metrics(model.tok2i, model.i2tok, field=dataloader.dataset.fields['trg'], bleu_to_file=bleu_to_file)
    predictions = []
    for i, batch in tqdm(enumerate(dataloader, 0), total=len(dataloader), disable=not log):
        scores, samples = common_eval.predict_batch(model, batch)
        predictions.extend(common_eval.convert_samples(samples, batch.trg[0], model.i2tok, model.tok2i))
        vmetrics.update(scores, samples, (batch.trg[0], None))
    model.train()
    ms = vmetrics.report('eval')
    return ms, predictions


def print_output(output_dict):
    print('SOURCE:\t%s' % ' '.join(output_dict['src_tokens']))
    print('ACTUAL:\t%s' % ' '.join(output_dict['gt_tokens']))
    print('PRED:\t%s' % ' '.join(output_dict['inorder_tokens']))
    print(output_dict['tree_string'])
    print('generation order:\t%s' % ' '.join(output_dict['genorder_tokens']))
    print()
