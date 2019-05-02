# Non-Monotonic Sequential Text Generation

PyTorch implementation of the paper:

[Non-Monotonic Sequential Text Generation](https://arxiv.org/pdf/1902.02192.pdf)\
Sean Welleck, Kiante Brantley, Hal Daume III, Kyunghyun Cho\
ICML 2019

We present code and data for training models described in the paper, and notebooks for evaluating pre-trained models.
## Installation

```bash
python setup.py develop
```

## Data
For downloading the datasets below, it may be helpful to use [gdown.pl](https://github.com/circulosmeos/gdown.pl).

#### Persona Chat (Unconditional Generation, Word-Reordering)
- [Google drive](https://drive.google.com/drive/folders/1XiLNkOsRaCZKpEmknOokq3Pi8jw8L5r6?usp=sharing)
- Put the `.jsonl` files into a directory `{PCHAT_DIR}`.

#### Machine Translation
- [Google drive](https://drive.google.com/open?id=1Stp56yZb6WsjEJhF9rxRegrNuV3ddziA)
- Unzip the dataset, e.g. to `/path/to/iwslt`. Then `{MT_DIR}` below will be `/path/to/iwslt/IWSLT/en-de/`.


## Using a Pretrained Model
You can use and evaluate pre-trained models in one of the provided notebooks: 

| Task | Models | Notebook |
| -------------   | --- | -------------  |
| **Word Reordering**| [Google drive](https://drive.google.com/file/d/1UX5_6E7vOiBzFoO0tCh5Pu5CzAWm2flE/view?usp=sharing) | [notebooks/word_reorder_eval.ipynb](notebooks/word_reorder_eval.ipynb)|
| **Unconditional Generation**|[Google drive](https://drive.google.com/file/d/1HmtxtzGG3tvQBk6tPtKn_OsMnA0LcjqX/view?usp=sharing)| [notebooks/unconditional_eval.ipynb](notebooks/unconditional_eval.ipynb)|
| **Translation (Transformer)**| [Google drive](https://drive.google.com/file/d/172Ir1oNvHBgnLO1hWqDeiAcBH5i6pfwi/view?usp=sharing) |[notebooks/translation_eval.ipynb](notebooks/translation_eval.ipynb)|

The word-reordering and translation notebooks reproduce the evaluation metrics (e.g. BLEU) in the paper. 

The unconditional notebook demos the models via interactive sampling and tree completion.

## Training 

First download and unzip [GloVe vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip) into a directory `{GLOVE_DIR}`.

### Word-Reordering
```bash
python tree_text_gen/binary/bagorder/train.py --glovepath {GLOVE_DIR}/glove.840B.300d.txt \
                                              --datadir {PCHAT_DIR} 
```

### Unconditional Generation
```bash
python tree_text_gen/binary/unconditional/train.py --glovepath {GLOVE_DIR}/glove.840B.300d.txt \
                                                   --datadir {PCHAT_DIR} 
```

### Machine Translation (Transformer)
```bash
python tree_text_gen/binary/translation/train_transformer.py --datadir {MT_DIR}
```
Use `--multigpu` for multi-GPU.

### Machine Translation (LSTM)
```bash
python tree_text_gen/binary/translation/train.py --datadir {MT_DIR} --model-type translation \
                                                 --beta-burnin 2 --beta-step 0.05 \
                                                 --self-teach-beta-step 0.05
```

By default these commands train policies with the annealed oracle. See `tree_text_gen/{bagorder, unconditional, translation}/args.py` for hyper-parameters and arguments.
