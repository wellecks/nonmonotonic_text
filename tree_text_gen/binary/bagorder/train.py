import os
import time
import argparse
import json
import gtimer as gt
from itertools import chain

import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader

import tree_text_gen.binary.common.samplers as samplers
from tree_text_gen.binary.common.util import setup, init_embeddings, get_optimizer, log_tensorboard
from tree_text_gen.binary.common.data import load_personachat, build_tok2i, SentenceDataset, inds2toks
from tree_text_gen.binary.common.oracle import Oracle, LeftToRightOracle
from tree_text_gen.binary.common.losses import sequential_set_no_stop_loss, sequential_set_loss
from tree_text_gen.binary.common.tree import build_tree, tree_to_text, print_tree
from tree_text_gen.binary.common.model import LSTMDecoder
from tree_text_gen.binary.common.encoder import BOWEncoder

from tree_text_gen.binary.bagorder.metrics import Metrics
from tree_text_gen.binary.bagorder.args import model_args

parser = argparse.ArgumentParser()
model_args(parser)
args = parser.parse_args()
args = setup(args)

# -- DATA
if args.dataset == 'personachat':
    train = load_personachat(os.path.join(args.datadir, 'personachat_all_sentences_train.jsonl'))
    valid = load_personachat(os.path.join(args.datadir, 'personachat_all_sentences_valid.jsonl'))
    test = load_personachat(os.path.join(args.datadir, 'personachat_all_sentences_test.jsonl'))
    tok2i = build_tok2i(list(chain.from_iterable([d['tokens'] for d in (train + valid)])))
    i2tok = {j: i for i, j in tok2i.items()}

args.n_classes = len(tok2i)

train = SentenceDataset(train, tok2i, max_tokens=args.max_tokens)
valid = SentenceDataset(valid, tok2i, max_tokens=args.max_tokens)
trainloader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=train.collate, drop_last=True)
validloader = DataLoader(valid, batch_size=args.batch_size, shuffle=True, collate_fn=valid.collate, drop_last=True)
print("%d train\t%d valid" % (len(train), len(valid)))


# -- Samplers and roll-in
rollin_sampler = samplers.initialize(args)

# -- MODEL
model_config = {
    'fc_dim':        args.fc_dim,
    'dec_lstm_dim':  args.dec_lstm_dim,
    'dec_n_layers':  args.dec_n_layers,
    'n_classes':     args.n_classes,
    'word_emb_dim':  300,  # glove
    'dropout':       args.dropout,
    'device':        str(args.device),
    'longest_label': 10,  # gets adjusted during training
    'share_inout_emb': args.share_inout_emb,
    'nograd_emb': args.nograd_emb,
    'batch_size': args.batch_size,
    'model_type': args.model_type,
    'aux_end': args.aux_end
}
encoder = BOWEncoder(model_config, tok2i)
model = eval(args.decoder)(model_config, tok2i, rollin_sampler, encoder).to(args.device)
init_embeddings(model.encoder.enc_emb, tok2i, args.glovepath)
init_embeddings(model.dec_emb, tok2i, args.glovepath)
print(model)


# -- oracle
oracle_flags = {}
if 'uniform' in args.oracle:
    Oracle = Oracle
elif 'leftright' in args.oracle:
    Oracle = LeftToRightOracle

# -- loss
loss_flags = {}
if args.aux_end:
    loss_fn = sequential_set_loss
else:
    loss_fn = sequential_set_no_stop_loss
loss_flags['self_teach_beta'] = args.self_teach_beta


# -- save things for eval time loading
with open(os.path.join(args.log_directory, 'model_config.json'), 'w') as f:
    json.dump(model_config, f)
with open(os.path.join(args.log_directory, 'tok2i.json'), 'w') as f:
    json.dump(tok2i, f)


# -- optimizer
optim_fn, optim_params = get_optimizer(args.optimizer)
optimizer = optim_fn(model.parameters(), **optim_params)


# -- globals
val_metric_best = -1e10
stop_training = False
lr = optim_params['lr']


def train_epoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    model.train()
    losses = []
    logs = []

    last_time = time.time()

    metrics = Metrics(tok2i, i2tok, bleu_to_file=False)
    for i, data in enumerate(trainloader, 0):
        # -- Actual Training
        gt.reset()
        xs, annots = data
        xs = xs.to(args.device)
        gt.stamp("load_data")

        oracle = Oracle(xs, model.n_classes, tok2i, i2tok, **oracle_flags)
        gt.stamp("create_oracle")
        max_steps = 2*xs.ne(tok2i['<p>']).sum(1).max()+1
        scores, samples, p_oracle = model.forward(xs=xs, oracle=oracle, max_steps=max_steps, return_p_oracle=True)
        gt.stamp("forward")
        loss = loss_fn(scores, samples, p_oracle, tok2i['<end>'], **loss_flags)
        gt.stamp("loss")

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        gt.stamp("backward")

        losses.append(loss.item())

        # -- Report metrics every `print_every` batches.
        if i % args.print_every == 0:
            # Training report; loss is averaged over the last `print_every` batches.
            metrics.update(scores, samples, data)
            gt.stamp("metrics.update")
            ms = metrics.report('train')
            ms['train/loss'] = round(np.mean(losses), 2)
            logs.append('{0} ; loss {1} ; sentence/s {2} ; {3} train {4} '.format(
                        i+1,
                        round(np.mean(losses), 2),
                        int(len(losses) * args.batch_size / (time.time() - last_time)),
                        args.eval_metric,
                        ms['train/%s' % args.eval_metric],
                        ))
            args.logstep += 1
            last_time = time.time()
            losses = []
            metrics.reset()

            # Validation report with a single batch.
            metrics.reset()
            model.eval()

            data = next(validloader.__iter__())
            scores, samples = predict_batch(data)
            model.train()
            metrics.update(scores, samples, data)
            vms = metrics.report('valid_batch')
            logs[-1] = logs[-1] + metrics.log(vms, 'valid_batch', ['bleu', 'avg_span', 'f1', 'em', 'depth_score'])
            metrics.reset()

            print_samples(samples, data)
            gt.stamp("validation_batch")

            log_tensorboard(ms, step=args.logstep)
            log_tensorboard(vms, step=args.logstep)
            print(logs[-1])
            print(gt.report(include_itrs=False, format_options={'itr_name_width': 30}))

        # -- Checkpointing
        if i % args.save_every == 0:
            print('saving checkpoint at epoch {0} batch {1}'.format(epoch, i))
            torch.save(model.state_dict(), os.path.join(args.log_directory, args.expr_name + '.checkpoint'))
            model_config['longest_label'] = model.longest_label
            with open(os.path.join(args.log_directory, 'model_config.json'), 'w') as f:
                json.dump(model_config, f)

    print('end : epoch {0} '.format(epoch))
    log_tensorboard({'lr': optimizer.param_groups[0]['lr']}, step=args.logstep)


def predict_batch(data):
    with torch.no_grad():
        xs, annots = data
        xs = xs.to(args.device)
        model.eval()
        scores, samples = model.forward(xs, oracle=None)
        model.train()
    return scores, samples


def print_samples(samples, data, n=min(args.batch_size, 5)):
    for i in range(n):
        tokens = inds2toks(i2tok, samples[i].cpu().tolist())
        root = build_tree(tokens)
        tokens, nodes = tree_to_text(root)
        tokens_levels = [(node.value, node.level) for node in nodes]
        gt_inds = [x for x in data[0][i].cpu().tolist() if x != tok2i['</s>'] and x != tok2i['<p>']]
        gt_tokens = inds2toks(i2tok, gt_inds)
        print('ACTUAL:\t%s' % ' '.join(gt_tokens))
        print('PRED:\t%s' % ' '.join(tokens))
        print(' '.join(str(x) for x in tokens_levels))
        print(print_tree(root))
        print()


def evaluate(epoch, dataloader, eval_type='valid', final_eval=False):
    global val_metric_best, lr, stop_training

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    vmetrics = Metrics(tok2i, i2tok)
    vmetrics.reset()
    model.eval()
    for i, data in enumerate(dataloader, 0):
        scores, samples = predict_batch(data)
        vmetrics.update(scores, samples, data)
    model.train()

    kind = eval_type if not final_eval else 'final_' + eval_type
    ms = vmetrics.report(kind)
    eval_metric = ms['%s/%s' % (kind, args.eval_metric)]
    metrics_to_log = ['bleu', 'avg_span', 'f1', 'em', 'depth_score']
    if final_eval:
        print('final: ' + vmetrics.log(ms, kind, metrics_to_log))
        log_tensorboard(ms, step=args.logstep)
    else:
        print(('valid (epoch %d): ' % epoch) + vmetrics.log(ms, kind, metrics_to_log))
        log_tensorboard(ms, step=args.logstep)

    if eval_type == 'valid' and epoch <= args.n_epochs:
        if eval_metric >= val_metric_best:
            print('saving model at epoch {0}'.format(epoch))
            torch.save(model.state_dict(), os.path.join(args.log_directory, args.expr_name))
            val_metric_best = eval_metric
        if epoch > 1 and epoch % args.lrshrink_nepochs == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / args.lrshrink
            print('Shrinking lr by : {0}. New lr = {1}'
                  .format(args.lrshrink, optimizer.param_groups[0]['lr']))
    return eval_metric


def adjust(epoch, sampler):
    if epoch <= args.beta_burnin:
        return
    if hasattr(sampler, 'beta'):
        sampler.beta = max(sampler.beta - args.beta_step, 0.0)
        log_tensorboard({'sampler.beta': sampler.beta}, step=args.logstep)
    if args.self_teach_beta_step > 0 and 'self_teach_beta' in loss_flags:
        loss_flags['self_teach_beta'] = max(loss_flags['self_teach_beta'] - args.self_teach_beta_step, 0.0)
        log_tensorboard({'self_teach_beta': loss_flags['self_teach_beta']}, step=args.logstep)


"""Train model"""
epoch = 1

while not stop_training and epoch <= args.n_epochs:
    train_epoch(epoch)
    adjust(epoch, model.sampler)
    evaluate(epoch, validloader, 'valid')
    epoch += 1

# Run best model on test set.
checkpoint = torch.load(os.path.join(args.log_directory, args.expr_name))
model.load_state_dict(checkpoint)
test = SentenceDataset(test, tok2i)
testloader = DataLoader(test, batch_size=args.batch_size, shuffle=True, collate_fn=valid.collate)

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, validloader, 'valid', True)
evaluate(0, testloader, 'test', True)
