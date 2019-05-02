from tqdm import tqdm
import json
import os
import torch as th
import tree_text_gen.binary.common.samplers as samplers
import tree_text_gen.binary.common.model as models
import tree_text_gen.binary.common.data as data
import tree_text_gen.binary.common.tree as tree_util
import tree_text_gen.binary.common.evaluate as common_eval
from tree_text_gen.binary.common.encoder import BOWEncoder
from tree_text_gen.binary.bagorder.metrics import Metrics

def load_model(expr_dir, expr_name, checkpoint, parent_lstm=False, decoder_class=None):
    model_file = os.path.join(expr_dir, expr_name + ('.checkpoint' if checkpoint else ''))
    tok2i = json.load(open(os.path.join(expr_dir, 'tok2i.json'), 'r'))
    config = json.load(open(os.path.join(expr_dir, 'model_config.json'), 'r'))
    config['device'] = th.device('cpu') if config['device'] == 'cpu' or not th.cuda.is_available() else th.device('cuda:0')
    config['model_type'] = config.get('model_type', 'bagorder')

    encoder = BOWEncoder(config, tok2i)
    sampler = samplers.GreedySampler()
    rollin_sampler = samplers.MixedRollin(0.0, sampler, sampler, 'trajectory')
    if parent_lstm:
        model = models.ParentLSTMDecoder(config, tok2i, rollin_sampler, encoder).to(config['device'])
    elif decoder_class is None:
        model = models.LSTMDecoder(config, tok2i, rollin_sampler, encoder).to(config['device'])
    else:
        model = decoder_class(config, tok2i, rollin_sampler, encoder).to(config['device'])
    if config['device'] == 'cpu' or not th.cuda.is_available():
        checkpoint = th.load(model_file, map_location='cpu')
    else:
        checkpoint = th.load(model_file)
    model.load_state_dict(checkpoint)
    i2tok = {j: i for i, j in tok2i.items()}
    model.tok2i = tok2i
    model.i2tok = i2tok
    return model


def eval_single(model, sentence):
    model.eval()
    xs = sentence.split()  # NOTE(wellecks): split tokenizer
    idxs = ([model.tok2i.get(x, model.tok2i['<unk>']) for x in xs] +
            [model.tok2i['</s>']])
    x = th.tensor([idxs], dtype=th.long, device=model.device)
    scores, preds = model.forward(x)
    raw_tokens = data.inds2toks(model.i2tok, preds.cpu().tolist()[0])
    model.train()

    root = tree_util.build_tree(raw_tokens)
    inorder_tokens, nodes = tree_util.tree_to_text(root)

    tokens_levels = [(node.value, node.level) for node in nodes]
    output = {'raw_tokens': raw_tokens,
              'tree_string': tree_util.print_tree(root),
              'inorder_tokens': inorder_tokens,
              'genorder_tokens': common_eval.get_genorder_tokens(raw_tokens),
              'token_levels': tokens_levels,
              'gt_tokens': xs}
    return output, x, scores, preds


def eval_dataset(model, dataloader, **kwargs):
    model.eval()
    vmetrics = Metrics(model.tok2i, model.i2tok)
    predictions = []
    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        scores, samples = common_eval.predict_batch(model, data)
        predictions.extend(common_eval.convert_samples(samples, data[0], model.i2tok, model.tok2i))
        vmetrics.update(scores, samples, data)
    model.train()
    ms = vmetrics.report('eval')
    return ms, predictions


def print_output(output_dict):
    print('ACTUAL:\t%s' % ' '.join(output_dict['gt_tokens']))
    print('PRED:\t%s' % ' '.join(output_dict['inorder_tokens']))
    print(output_dict['tree_string'])
    print('generation order:\t%s' % ' '.join(output_dict['genorder_tokens']))
    print()

