import json
import nltk
import os
import torch as th
from tqdm import tqdm
from multiprocessing import Pool
from nltk.translate.bleu_score import SmoothingFunction

import tree_text_gen.binary.common.samplers as samplers
import tree_text_gen.binary.common.model as models
import tree_text_gen.binary.common.evaluate as common_eval
from tree_text_gen.binary.unconditional.metrics import Metrics


def load_model(expr_dir, expr_name, checkpoint, topk_sampler=-1):
    model_file = os.path.join(expr_dir, expr_name + ('.checkpoint' if checkpoint else ''))
    tok2i = json.load(open(os.path.join(expr_dir, 'tok2i.json'), 'r'))
    config = json.load(open(os.path.join(expr_dir, 'model_config.json'), 'r'))
    config['device'] = th.device('cpu') if config['device'] == 'cpu' or not th.cuda.is_available() else th.device('cuda:0')
    config['model_type'] = config.get('model_type', 'unconditional')

    if topk_sampler == -1:
        sampler = samplers.StochasticSampler()
    else:
        sampler = samplers.TopkSampler(topk_sampler, config['device'])
    rollin_sampler = samplers.MixedRollin(0.0, sampler, sampler, 'trajectory')
    model = models.LSTMDecoder(config, tok2i, rollin_sampler, encoder=None).to(config['device'])
    if config['device'] == 'cpu' or not th.cuda.is_available():
        checkpoint = th.load(model_file, map_location='cpu')
    else:
        checkpoint = th.load(model_file)
    model.load_state_dict(checkpoint)
    i2tok = {j: i for i, j in tok2i.items()}
    model.tok2i = tok2i
    model.i2tok = i2tok
    return model


def sample_with_prefix(model, tree_prefix_tokens, n=10):
    idxs = ([model.tok2i['<s>']] + [model.tok2i.get(x, model.tok2i['<unk>']) for x in tree_prefix_tokens])
    xs = th.tensor([idxs], dtype=th.long, device=model.device)
    xs = xs.expand(n, xs.size(1)).contiguous()

    model.eval()
    B = xs.size(0)
    hidden = (th.zeros(model.dec_n_layers, B, model.dec_lstm_dim, device=model.device),
              th.zeros(model.dec_n_layers, B, model.dec_lstm_dim, device=model.device))
    scores = []
    samples = []
    model.sampler.reset(bsz=B)

    with th.no_grad():
        # forward the prefix
        for t in range(xs.size(1)):
            xt = xs[:, t].unsqueeze(1)
            score_t, _, hidden = model.forward_decode(xt, hidden, None)
            scores.append(score_t)

        # add the non-start tokens
        for t in range(1, xs.size(1)):
            samples.append(xs[:, t].detach().unsqueeze(1))

        # start sampling
        for t in range(model.longest_label - xs.size(1) - 1):
            xt = model.sampler(score_t, oracle=None, training=False)
            score_t, _, hidden = model.forward_decode(xt, hidden, None)
            scores.append(score_t)
            samples.append(xt.detach())

    scores = th.cat(scores, 1)
    samples = th.cat(samples, 1)
    output = common_eval.convert_samples(samples, None, model.i2tok, model.tok2i)
    return output, scores, samples


def predict_batch(model, batch_size=32):
    with th.no_grad():
        model.eval()
        scores, samples = model.forward(num_samples=batch_size, oracle=None)
        model.train()
    return scores, samples


def sample(model, n=1):
    model.eval()
    scores, samples = predict_batch(model, batch_size=n)
    model.train()
    outputs = common_eval.convert_samples(samples, None, model.i2tok, model.tok2i)
    return outputs


def calc_bleu(reference, hypothesis, weight):
    return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                   smoothing_function=SmoothingFunction().method1)


def get_bleu_parallel(ngram, reference, generated, self_bleu, cpu_limit=None):
    weight = tuple((1. / ngram for _ in range(ngram)))
    count = min(cpu_limit, os.cpu_count()) if cpu_limit is not None else os.cpu_count()
    pool = Pool(count)
    result = list()
    for i, hypothesis in enumerate(generated):
        if self_bleu:
            reference_ = generated[:i] + generated[i+1:]
        else:
            reference_ = reference
        result.append(pool.apply_async(calc_bleu, args=(reference_, hypothesis, weight)))
    score = 0.0
    cnt = 0
    for i in tqdm(result, total=len(result)):
        score += i.get()
        cnt += 1
    pool.close()
    pool.join()
    return score / cnt


def bleu_eval(model, reference_sentences, n_samples=1000, ks=(10, 100, 1000, -1), bleu_ns=(2,3,4,5), sample_reference=False, self_bleu=False):
    # ref: https://github.com/geek-ai/Texygen/blob/3104e22ac75f3cc2070da2bf5e2da6d2bef149ad/utils/metrics/Bleu.py
    if sample_reference:
        import random
        reference_sentences = random.sample(reference_sentences, n_samples)
    all_metrics = {}
    all_samples = {}
    for k in ks:
        if k == -1:
            model.sampler.eval_sampler = samplers.StochasticSampler()
        else:
            model.sampler.eval_sampler = samplers.TopkSampler(k, model.device)

        metrics = {}
        batch_size = min(n_samples, 100)
        n_batches = n_samples // batch_size
        samples = []
        model.eval()
        for _ in range(n_batches):
            out = sample(model, batch_size)
            for x in out:
                samples.append(x['inorder_tokens'])

        model.train()
        for n in bleu_ns:
            metrics['bleu-%d' % n] = get_bleu_parallel(n, reference_sentences, samples, self_bleu=False)
            if self_bleu:
                metrics['self-bleu-%d' % n] = get_bleu_parallel(n, reference_sentences, samples, self_bleu=True)

        all_metrics[k] = metrics
        all_samples[k] = samples
    return all_metrics, all_samples


def eval_dataset(model, dataloader):
    model.eval()
    vmetrics = Metrics(model.tok2i, model.i2tok)
    predictions = []
    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        scores, samples = predict_batch(model, data)
        predictions.extend(common_eval.convert_samples(samples, data[0], model.i2tok, model.tok2i))
        vmetrics.update(scores, samples, data)
    model.train()
    ms = vmetrics.report('eval')
    return ms, predictions


def print_output(output_dict):
    print('PRED:\t%s' % ' '.join(output_dict['inorder_tokens']))
    print(output_dict['tree_string'])
    print('generation order:\t%s' % ' '.join(output_dict['genorder_tokens']))
    print()
