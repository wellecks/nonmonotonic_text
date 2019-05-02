"""Train script for LSTM"""
import os
import time
import argparse
import json
import logging
import copy
import gtimer as gt
import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torchtext import data

import tree_text_gen.binary.common.samplers as samplers
import tree_text_gen.binary.common.constants as constants
from tree_text_gen.binary.common.util import setup, get_optimizer, log_tensorboard
from tree_text_gen.binary.common.losses import sequential_set_no_stop_loss, sequential_set_loss
from tree_text_gen.binary.common.tree import build_tree, tree_to_text, print_tree
from tree_text_gen.binary.common.oracle import Oracle, LeftToRightOracle
from tree_text_gen.binary.common.model import LSTMDecoder
from tree_text_gen.binary.common.encoder import LSTMEncoder

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

# -- DATA
train_data, dev_data, test_data, SRC, TRG = load_iwslt(args)
tok2i, i2tok, SRC, TRG = load_iwslt_vocab(args, SRC, TRG, args.data_prefix)
SRC = copy.deepcopy(SRC)
for data_ in [train_data, dev_data, test_data]:
    if not data_ is None:
        data_.fields['src'] = SRC

# NOTE: Only have to sort by source because the encoder uses rnn.pack_padded_sequence and the decoder does not
sort_key = lambda x: len(x.src)
trainloader = data.BucketIterator(dataset=train_data, batch_size=args.batch_size, device=args.device, train=True, repeat=False, shuffle=True, sort_key=sort_key, sort_within_batch=True) if not train_data is None else None
validloader = data.BucketIterator(dataset=dev_data, batch_size=args.batch_size, device=args.device, train=False, repeat=False, shuffle=True, sort_key=sort_key, sort_within_batch=True) if not dev_data is None else None
testloader = data.BucketIterator(dataset=test_data, batch_size=args.batch_size, device=args.device, train=False, repeat=False, shuffle=False, sort_key=sort_key, sort_within_batch=True) if not test_data is None else None

# -- MODEL
if args.model_dir is None:
    model_config = {
        'fc_dim':        args.fc_dim,
        'dec_lstm_dim':  args.dec_lstm_dim,
        'enc_lstm_dim': args.enc_lstm_dim,
        'dec_n_layers':  args.dec_n_layers,
        'n_classes':     len(tok2i),
        'word_emb_dim':  256,  # glove
        'dropout':       args.dropout,
        'device':        str(args.device),
        'longest_label': 10,  # gets adjusted during training
        'share_inout_emb': args.share_inout_emb,
        'nograd_emb': args.nograd_emb,
        'num_dir_enc': 2,
        'vocab_size': args.vocab_size,
        'src': args.src,
        'trg': args.trg,
        'enc_n_layers': args.num_layers_enc,
        'batch_size': args.batch_size,
        'model_type': args.model_type,
        'aux_end': args.aux_end
    }

args.n_classes = len(TRG.vocab.stoi)
rollin_sampler = samplers.initialize(args)

encoder = LSTMEncoder(model_config, tok2i)
model = eval(args.decoder)(model_config, tok2i, rollin_sampler, encoder).to(args.device)

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

    metrics = Metrics(tok2i, i2tok, field=TRG)
    for i, batch in enumerate(trainloader):
        # -- Actual Training
        gt.reset()
        gt.stamp("load_data")

        oracle = Oracle(batch.trg[0].detach(), model.n_classes, tok2i, i2tok, **oracle_flags)
        gt.stamp("create_oracle")
        max_steps = 2*batch.trg[0].detach().ne(tok2i[constants.PAD_WORD]).sum(1).max()+1
        scores, samples, p_oracle = model.forward(xs=batch.src, oracle=oracle, max_steps=max_steps, num_samples=len(batch),
                                                  return_p_oracle=True)
        gt.stamp("forward")
        loss = loss_fn(scores, samples, p_oracle, end_idx=tok2i['<end>'], **loss_flags)
        gt.stamp("loss")

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        gt.stamp("backward")

        losses.append(loss.item())

        # -- Report metrics every `print_every` batches.
        if i % args.print_every == 0:
            # Only compute training metrics once here for efficiency.
            metrics.update(scores, samples, (batch.trg[0], None), kind='train')
            gt.stamp("metrics.update")
            # Training report computed over the last `print_every` batches.
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
        if i % args.save_every == 0:
            print('saving checkpoint at epoch {0} batch {1}'.format(epoch, i))
            print(os.path.join(args.log_directory, args.expr_name + '.checkpoint'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_param': args.optimizer,
                'loss': loss.item()
            }, os.path.join(args.log_directory, args.expr_name + '.checkpoint'))

            model_config['longest_label'] = model.longest_label
            with open(os.path.join(args.log_directory, 'model_config.json'), 'w') as f:
                json.dump(model_config, f)

    print('end : epoch {0} '.format(epoch))
    log_tensorboard({'lr': optimizer.param_groups[0]['lr']}, step=args.logstep)

def predict_batch(batch):
    with torch.no_grad():
        model.eval()
        scores, samples = model.forward(xs=batch.src, oracle=None, num_samples=len(batch))
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
print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, validloader, 'valid', True)
evaluate(0, testloader, 'test', True)
