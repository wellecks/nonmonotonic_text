import re
import os
import datetime
import numpy as np
import tensorboard_logger as logger
import torch.nn as nn
from pprint import pprint
import inspect
from torch import optim
import torch
from tqdm import tqdm


# --- Pytorch
def masked_softmax(vec, mask, dim=1, epsilon=1e-40, alpha=0.):
    exps = torch.exp(vec)
    masked_exps = exps * mask.float() + alpha
    masked_sums = torch.clamp(masked_exps.sum(dim, keepdim=True), min=epsilon)
    ps = masked_exps / masked_sums
    return ps


# --- Misc. Utilities
def get_optimizer(s):
    """ Parse optimizer parameters. Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DataParallel(nn.DataParallel):
    """allows model.attr instead of having to use model.module.attr"""
    def __getattr__(self, item):
        if item != 'module':
            return getattr(self.module, item)
        return super().__getattr__(item)


def init_embeddings(emb_layer, tok2ind, pretrained_path):
    n = 0
    seen = set()
    with open(pretrained_path) as f:
        for line in tqdm(f, total=2196017):  # Glove
            tok, vec = line.split(' ', 1)
            if tok in tok2ind:
                emb_layer.weight.data[tok2ind[tok]] = torch.tensor(list(map(float, vec.split())))
                n += 1
                seen.add(tok)
    for tok in tok2ind:
        if tok not in seen:
            emb_layer.weight.data[tok2ind[tok]] = torch.from_numpy(np.random.standard_normal(300))
    print('Found {0}(/{1}) words with pretrained vectors'.format(n, len(tok2ind)))


# --- Experiment Utilities
def setup_tensorboard(opts):
    """Creates a logging directory and configures a tensorboard logger."""
    base_dir = os.path.join(opts['log_base_dir'], opts['expr_name'])
    log_directory = date_filename(base_dir) + '__' + opts['expr_name']
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    try:
        logger.configure(log_directory)
    except ValueError:
        pass
    return log_directory


def log_tensorboard(values_dict, step):
    for k, v in values_dict.items():
        if isinstance(v, int) or isinstance(v, float):
            logger.log_value(k, v, step)


def setup(args, log=True, no_file_creation=False):
    """Perform boilerplate experiment setup steps, returning a dictionary of config options."""
    # set gpu device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        args.device = torch.device('cuda:%d' % args.gpu)
    else:
        args.device = torch.device('cpu')

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # logging and directory
    opts = args.__dict__.copy()
    if not no_file_creation:
        log_directory = setup_tensorboard(opts)
        opts['log_directory'] = log_directory
    else:
        opts['log_directory'] = ''
    opts['logger'] = logger
    opts['logstep'] = 1
    if log:
        pprint(opts)
    opts = dotdict(opts)
    return opts


def mkdir_p(path, log=True):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    if log:
        print('Created directory %s' % path)


def date_filename(base_dir='./'):
    dt = datetime.datetime.now()
    return os.path.join(base_dir, '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second))

