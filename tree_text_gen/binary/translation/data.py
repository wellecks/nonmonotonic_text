import os
import torch
from torchtext import data
from torchtext import vocab
from pathlib import Path
import tree_text_gen.binary.common.constants as constants
from collections import defaultdict


class NormalField(data.Field):
    def reverse(self, batch, unbpe=True):
        if not self.batch_first:
            batch.t_()

        if not isinstance(batch, list):
            with torch.cuda.device_of(batch):
                batch = batch.tolist()

        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch] # de-numericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch] # trim past first eos
        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        if unbpe:
            batch = [" ".join(filter(filter_special, ex)).replace("@@ <end>"," <end>").replace("@@ ", "")
                     for ex in batch]
        else:
            batch = [" ".join(filter(filter_special, ex)) for ex in batch]
        return batch


class TranslationDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, prefix='', **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))
        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path, exts, fields, train='train', validation='val', test='test', **kwargs):
        train_data = None if train is None else cls(os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


def load_iwslt(args):
    TRG = NormalField(init_token=constants.BOS_WORD, eos_token=constants.EOS_WORD,
                      pad_token=constants.PAD_WORD, unk_token=constants.UNK_WORD,
                      include_lengths=True, batch_first=True)
    SRC = NormalField(batch_first=True) if not args.share_vocab else TRG
    data_prefix = Path(args.datadir)
    args.data_prefix = data_prefix
    length_filter = lambda x: (len(vars(x)['src']) <= args.max_len_src - 2
                               and len(vars(x)['trg']) <= args.max_len_trg - 2
                               and len(vars(x)['src']) >= args.min_len
                               and len(vars(x)['trg']) >= args.min_len)

    # -- Train
    train_dir = "train"
    train_data = TranslationDataset(
        path=str(data_prefix / train_dir / 'train.tags.en-de.bpe'),
        exts=('.de', '.en'), fields=(SRC, TRG), prefix='normal',
        filter_pred=length_filter if args.mode in ["train"] else None)

    # -- Dev
    dev_dir = "dev"
    dev_file = "valid.en-de.bpe"
    if args.mode == "test" and args.decode_which > 0:
        dev_dir = "dev_split"
        dev_file += ".{}".format(args.decode_which)
    dev_data = TranslationDataset(path=str(data_prefix / dev_dir / dev_file),
                                  exts=('.de', '.en'), fields=(SRC, TRG), prefix='normal')

    # -- Test
    test_dir = "test"
    test_file = "test.bpe"
    if args.mode == "test" and args.decode_which > 0:
        test_dir = "test_split"
        test_file += ".{}".format(args.decode_which)
    path = str(data_prefix / test_dir / test_file)
    if os.path.exists(path + '.de'):
        test_data = TranslationDataset(path=path,
                                       exts=('.de', '.en'), fields=(SRC, TRG), prefix='normal')
    else:
        print("WARNING: Test set (%s) not available, using dev data as test." % path)
        test_data = dev_data
    return train_data, dev_data, test_data, SRC, TRG


def load_iwslt_vocab(args, SRC, TRG, data_prefix):
    vocab_path = data_prefix / 'vocab' / '{}-{}_{}_{}.pt'.format(args.src, args.trg,
                                                                 args.vocab_size,
                                                                 'shared' if args.share_vocab else '')
    if args.load_vocab and vocab_path.exists():
        src_vocab, trg_vocab = torch.load(str(vocab_path))
        SRC.vocab = src_vocab
        TRG.vocab = trg_vocab

    # SRC and TRG should point to the same Object
    if SRC.vocab.itos[0] == '<unk>' and SRC.vocab.itos[1] == '<pad>' and \
                    SRC.vocab.itos[2] == '<init>' and SRC.vocab.itos[3] == '<eos>' and \
                    SRC.vocab.itos[4] == ',' and SRC is TRG:
        # Replace the special tokens with this project's version.
        SRC.vocab.itos[1] = constants.PAD_WORD
        SRC.vocab.itos[2] = constants.BOS_WORD
        SRC.vocab.itos[3] = constants.EOS_WORD
        SRC.vocab.itos[4] = constants.NODE_END
        SRC.vocab.itos.append(',')
        SRC.vocab.stoi = defaultdict(vocab._default_unk_index)
        SRC.vocab.stoi.update({tok: i for i, tok in enumerate(SRC.vocab.itos)})
    else:
        raise Exception("Vocab token replacement is not working!")
    tok2i = TRG.vocab.stoi
    i2tok = TRG.vocab.itos
    i2tok = {v: k for v, k in enumerate(i2tok)}
    return tok2i, i2tok, SRC, TRG


