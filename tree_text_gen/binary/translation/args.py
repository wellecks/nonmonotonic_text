def common_args(parser):
    # data, logging, saving, etc.
    parser.add_argument("--log-base-dir", type=str, default='output/', help="Base directory for output")
    parser.add_argument("--expr-name", type=str, default="ttg", help="Name appended to experiment outputs")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--print-batch-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--datadir", default='/home/sw1986/datasets/iwslt/IWSLT/en-de/')

    # dataset-specific
    parser.add_argument("--max-tokens", type=int, default=-1)

    # training
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lrshrink", type=float, default=2.0, help="shrink factor for lr")
    parser.add_argument("--lrshrink-nepochs", type=int, default=20, help="epoch interval for lr shrink")
    parser.add_argument("--max-norm", type=float, default=1.0, help="max norm (grad clipping)")
    parser.add_argument("--eval-metric", default='bleu', help="metric used for model saving, early stopping, etc.")

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.001", help="adam or sgd,lr=0.1")

    # model
    parser.add_argument("--dec-lstm-dim", type=int, default=512)
    parser.add_argument("--enc-lstm-dim", type=int, default=512)
    parser.add_argument("--dec-n-layers", type=int, default=1)
    parser.add_argument("--fc-dim", type=int, default=512)
    parser.add_argument("--share_inout_emb", type=lambda x: (str(x).lower() == 'true'), default=True, help='use embedding weights for RNN output-to-scores')
    parser.add_argument("--nograd_emb", type=lambda x: (str(x).lower() == 'true'), default=False, help="don't update embedding weights when True")
    parser.add_argument("--model-type", choices=['translation', 'transformer'], default='transformer')
    parser.add_argument("--decoder", choices=['LSTMDecoder'], default='LSTMDecoder')
    parser.add_argument("--aux-end", type=lambda x: (str(x).lower() == 'true'), default=False)

    # transformer
    parser.add_argument("--transformer-config", choices=['small'], default='small')
    parser.add_argument("--transformer-auxiliary-end", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--transformer-tree-encoding", type=lambda x: (str(x).lower() == 'true'), default=False)

    # oracle
    parser.add_argument("--oracle", choices=['uniform', 'leftright'], default='uniform')
    parser.add_argument("--rollin-beta", type=float, default=1.0, help="probability of using oracle for a full roll-in")
    parser.add_argument("--beta-step", type=float, default=0.05, help="per-epoch decrease of rollin-beta")
    parser.add_argument("--beta-burnin", type=int, default=2, help="number of epochs before we start to decrease beta")
    parser.add_argument("--beta-min", type=float, default=0.00)

    # loss
    parser.add_argument("--loss", choices=['multiset'], default='multiset')
    parser.add_argument("--importance-weight", type=lambda x: (str(x).lower() == 'true'), default=False, help='adjust loss based on rollin sampling distribution (see losses.py)')
    parser.add_argument("--self-teach-beta-step", type=float, default=0.05, help='experimental (see losses.py)')
    parser.add_argument("--self-teach-beta", type=float, default=1.0)
    parser.add_argument("--self-teach-beta-min", type=float, default=0.0)

    # samplers & rollin
    parser.add_argument("--rollin", choices=["mixed"], default='mixed', help='learned rollin via --rollin-beta 0.0')
    parser.add_argument("--rollin-mix-type", choices=["trajectory", "state"], default="state", help="mix rollin at the trajectory or state level")
    parser.add_argument("--training-sampler", choices=["greedy", "stochastic", "policy_correct_greedy", "policy_correct_stochastic"], default="policy_correct_greedy")
    parser.add_argument("--eval-sampler", choices=["greedy", "stochastic"], default="greedy")

    # gpu
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")

    parser.add_argument("--src", type=str, default="de")
    parser.add_argument("--trg", type=str, default="en")

    parser.add_argument("--max_len_src", type=int, default=50)
    parser.add_argument("--max_len_trg", type=int, default=50)
    parser.add_argument("--max_len_dec", type=int, default=300)
    parser.add_argument("--min_len", type=int, default=1)

    parser.add_argument("--num_layers_enc", type=int, default=1)

    # other
    parser.add_argument('--mode',type=str, default='train',  choices=['train', 'test'])
    parser.add_argument('--load_vocab', action='store_true', default=True, help='load a pre-computed vocabulary')
    parser.add_argument('--vocab_size', type=int, default=40000,  help='limit the train set sentences to this many tokens')
    parser.add_argument('--share_vocab',  action='store_true', default=True, help='share vocabulary between src and target')
    parser.add_argument('--model_dir', type=str, default=None, help='Location of the model file. default is None')

