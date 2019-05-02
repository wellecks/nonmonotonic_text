import os
import time
import argparse
import json
import logging
import copy
import gtimer as gt
import numpy as np
import pickle

import torch
from torch.nn.utils import clip_grad_norm_
from torchtext import data

import tree_text_gen.binary.common.samplers as samplers
from tree_text_gen.binary.common.util import setup, get_optimizer, log_tensorboard, DataParallel
from tree_text_gen.binary.common.losses import sequential_set_loss, sequential_set_no_stop_loss
from tree_text_gen.binary.common.tree import build_tree, tree_to_text, print_tree
from tree_text_gen.binary.common.oracle import Oracle, LeftToRightOracle
import tree_text_gen.binary.common.trajectory_sampler as buffer
import tree_text_gen.binary.common.transformer as transformer

from tree_text_gen.binary.translation.args import common_args
from tree_text_gen.binary.translation.metrics import Metrics
from tree_text_gen.binary.translation.data import load_iwslt, load_iwslt_vocab

parser = argparse.ArgumentParser()
common_args(parser)
args = parser.parse_args()
args = setup(args)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
args.logger = logger
args.device = torch.device('cuda')

# -- DATA
train_data, dev_data, test_data, SRC, TRG = load_iwslt(args)
tok2i, i2tok, SRC, TRG = load_iwslt_vocab(args, SRC, TRG, args.data_prefix)
SRC = copy.deepcopy(SRC)
for data_ in [train_data, dev_data, test_data]:
    if not data_ is None:
        data_.fields['src'] = SRC

sort_key = lambda x: len(x.src)
trainloader = data.BucketIterator(dataset=train_data, batch_size=args.batch_size, device=args.device, train=True, repeat=False, shuffle=True, sort_key=sort_key, sort_within_batch=True) if not train_data is None else None
validloader = data.BucketIterator(dataset=dev_data, batch_size=args.batch_size, device=args.device, train=False, repeat=False, shuffle=True, sort_key=sort_key, sort_within_batch=True) if not dev_data is None else None
testloader = data.BucketIterator(dataset=test_data, batch_size=args.batch_size, device=args.device, train=False, repeat=False, shuffle=False, sort_key=sort_key, sort_within_batch=True) if not test_data is None else None

args.n_classes = len(TRG.vocab.stoi)


# -- loss
loss_flags = {}
if 'multiset' in args.loss:
    loss_fn = sequential_set_loss
    if not args.transformer_auxiliary_end:
        loss_fn = sequential_set_no_stop_loss
    loss_flags['self_teach_beta'] = float(args.self_teach_beta)


# -- model
model_config = transformer.SmallConfig(args.device)
model_config.auxiliary_end = args.transformer_auxiliary_end
model_config.tree_encoding = args.tree_encoding

model = transformer.Transformer(model_config, tok2i, tok2i, loss_fn)
model.tok2i = tok2i
model.i2tok = i2tok

model = DataParallel(model)
model.cuda()

if args.model_dir is not None:
    expr_name = args.model_dir.split('__')[-1]
    model_file = os.path.join(args.model_dir, expr_name + '.checkpoint')
    model_config = pickle.load(open(os.path.join(args.model_dir, 'model_config.pkl'), 'rb'))
    model_checkpoint = torch.load(model_file)
    model.load_state_dict(model_checkpoint['model_state_dict'])

print(model)

# -- oracle
oracle_flags = {}
if 'uniform' in args.oracle:
    Oracle = Oracle
elif 'leftright' in args.oracle:
    Oracle = LeftToRightOracle


# -- save things for eval time loading
with open(os.path.join(args.log_directory, 'model_config.pkl'), 'wb') as f:
    pickle.dump(model_config, f)
with open(os.path.join(args.log_directory, 'tok2i.json'), 'w') as f:
    json.dump(tok2i, f)


# -- optimizer
if args.model_dir is None:
    optim_fn, optim_params = get_optimizer(args.optimizer)
    optimizer = optim_fn(model.parameters(), **optim_params)
else:
    optim_fn, optim_params = get_optimizer(model_checkpoint['optimizer_param'])
    optimizer = optim_fn(model.parameters(), **optim_params)
    # optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])


# -- globals
val_metric_best = -1e10
stop_training = False
lr = optim_params['lr']

updates_per_epoch = 1000
oracle_samples_only = False

def train_epoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    model.train()
    losses = []
    logs = []
    sample_avgs = []
    update_avgs = []

    last_time = time.time()
    metrics = Metrics(tok2i, i2tok, field=TRG)

    trajectory_sampler = buffer.TrajectorySampler(trainloader)
    n_updates = 0
    oracle_samples_only = args.rollin_beta == 1.0
    while n_updates < updates_per_epoch:
        gt.reset()
        if oracle_samples_only:
            start = time.time()
            trajectory = trajectory_sampler.get_oracle_trajectory(model, Oracle, oracle_flags=oracle_flags)
            sample_time = (time.time() - start)
            start = time.time()
            loss = trajectory_sampler.get_loss(model, trajectory, loss_flags)
            update_time = (time.time() - start)
        else:
            start = time.time()
            loss = trajectory_sampler.get_mixed_trajectory_loss(model, Oracle,
                                                                oracle_flags=oracle_flags,
                                                                beta=args.rollin_beta,
                                                                loss_flags=loss_flags)
            sample_time = 0
            update_time = (time.time() - start)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), args.max_norm)
        losses.append(loss.item())
        optimizer.step()
        n_updates += 1

        sample_avgs.append(sample_time)
        update_avgs.append(update_time)

        gt.stamp("buffer updates")

        if n_updates % 20 == 0:
            print("%d|%d\t%.3f\tSample: %.3fs\tUpdate: %.3fs" % (epoch, n_updates, round(np.mean(losses), 3), np.mean(sample_avgs), np.mean(update_avgs)))
            log_tensorboard({'sample_avgs': np.mean(sample_avgs),
                             'update_avgs': np.mean(update_avgs)}, step=args.logstep)
            sample_avgs = []
            update_avgs = []

        # -- Report metrics every `print_every` batches.
        if n_updates % args.print_every == 0:
            gt.stamp("report")
            # Training report computed over the last `print_every` batches.
            ms = metrics.report('train')
            ms['train/loss'] = round(np.mean(losses), 2)
            logs.append('{0} ; loss {1} ; sentence/s {2} ; {3} train {4} '.format(
                        epoch,
                        round(np.mean(losses), 2),
                        int(len(losses) * args.batch_size / (time.time() - last_time)),
                        args.eval_metric,
                        ms.get('train/%s' % args.eval_metric, 0.0),
                        ))
            args.logstep += 1
            last_time = time.time()
            losses = []
            metrics.reset()

            # -- Validation report with a single batch.
            metrics.reset()
            model.eval()
            batch = next(iter(validloader))
            scores, samples = predict_batch(batch)
            model.train()
            metrics.update(scores, samples, (batch.trg[0], None))
            vms = metrics.report('valid_batch')
            logs[-1] = logs[-1] + metrics.log(vms, 'valid_batch', ['bleu', 'avg_span', 'f1', 'em', 'depth_score'])
            metrics.reset()

            print_samples(samples, (batch.trg[0], None), n=len(batch))
            gt.stamp("validation_batch")

            log_tensorboard(ms, step=args.logstep)
            log_tensorboard(vms, step=args.logstep)
            print(logs[-1])
            print(gt.report(include_itrs=False, format_options={'itr_name_width': 30}))

        # -- Checkpointing
        if n_updates % args.save_every == 0:
            print('saving checkpoint at epoch {0} batch {1}'.format(epoch, i))
            print(os.path.join(args.log_directory, args.expr_name + '.checkpoint'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_param': args.optimizer,
                'loss': loss.item()
            }, os.path.join(args.log_directory, args.expr_name + '.checkpoint'))

            model_config.longest_label = model.longest_label
            with open(os.path.join(args.log_directory, 'model_config.pkl'), 'wb') as f:
                pickle.dump(model_config, f)

    print('end : epoch {0} '.format(epoch))
    log_tensorboard({'lr': optimizer.param_groups[0]['lr']}, step=args.logstep)

def predict_batch(batch):
    with torch.no_grad():
        model.eval()
        scores, samples = model(xs=batch.src, ys=None, p_oracle=None, num_samples=len(batch))
        model.train()
    return scores, samples


def print_samples(samples, data, n=None):
    n = min(n, 5)
    for i in range(n):
        tokensa = [x.split() for x in TRG.reverse(samples[i].unsqueeze(0), unbpe=True)][0]
        root = build_tree(tokensa)
        tokens, nodes = tree_to_text(root)
        gt_tokens = [x.split() for x in TRG.reverse(data[0][i:i + 1].cpu(), unbpe=True)][0]
        print('ACTUAL:\t%s' % ' '.join(gt_tokens))
        print('PRED:\t%s' % ' '.join(tokens))
        print(print_tree(root))
        print()


def evaluate(epoch, dataloader, eval_type='valid', final_eval=False):
    global val_metric_best, lr, stop_training

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    vmetrics = Metrics(tok2i, i2tok, field=TRG)
    vmetrics.reset()
    model.eval()
    for i, batch in enumerate(dataloader, 0):
        scores, samples = predict_batch(batch)
        vmetrics.update(scores, samples, (batch.trg[0], None))
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


def adjust(epoch):
    if epoch <= args.beta_burnin:
        return
    args.rollin_beta = max(args.rollin_beta - args.beta_step, args.beta_min)
    log_tensorboard({'sampler.beta': args.rollin_beta}, step=args.logstep)
    if args.self_teach_beta_step > 0 and 'self_teach_beta' in loss_flags:
        loss_flags['self_teach_beta'] = max(loss_flags['self_teach_beta'] - args.self_teach_beta_step, 0.0)
        log_tensorboard({'self_teach_beta': loss_flags['self_teach_beta']}, step=args.logstep)


"""Train model"""
if args.model_dir is None:
    epoch = 1
else:
    epoch = model_checkpoint['epoch']

while not stop_training and epoch <= args.n_epochs:
    train_epoch(epoch)
    adjust(epoch)
    evaluate(epoch, validloader, 'valid')
    epoch += 1

# Run best model on test set.
checkpoint = torch.load(os.path.join(args.log_directory, args.expr_name))
model.load_state_dict(checkpoint)
print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, validloader, 'valid', True)
evaluate(0, testloader, 'test', True)
