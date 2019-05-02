
def model_args(parser):
    # data, logging, saving, etc.
    parser.add_argument("--glovepath", type=str, default='/Users/wellecks/own_files/datasets/glove/glove.840B.300d.txt')
    parser.add_argument("--dataset", choices=['personachat'], default='personachat')
    parser.add_argument("--datadir", type=str, default='/Users/wellecks/own_files/datasets/personachat')
    parser.add_argument("--log-base-dir", type=str, default='output/', help="Base directory for output")
    parser.add_argument("--expr-name", type=str, default="unconditional", help="Name appended to experiment outputs")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--print-batch-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)

     # dataset-specific
    parser.add_argument("--max-tokens", type=int, default=-1)

    # training
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.001", help="adam or sgd,lr=0.1")
    parser.add_argument("--lrshrink", type=float, default=2.0, help="shrink factor for lr")
    parser.add_argument("--lrshrink-nepochs", type=int, default=20, help="epoch interval for lr shrink")
    parser.add_argument("--max-norm", type=float, default=1., help="max norm (grad clipping)")

    # model
    parser.add_argument("--dec-lstm-dim", type=int, default=1024)
    parser.add_argument("--dec-n-layers", type=int, default=2)
    parser.add_argument("--fc-dim", type=int, default=512)
    parser.add_argument("--share_inout_emb", type=bool, default=True, help='use embedding weights for RNN output-to-scores')
    parser.add_argument("--nograd_emb", type=bool, default=False)
    parser.add_argument("--model-type", choices=['unconditional'], default='unconditional')
    parser.add_argument("--aux-end", type=lambda x: (str(x).lower() == 'true'), default=False)

    # oracle
    parser.add_argument("--oracle", choices=['uniform', 'leftright'], default='uniform')
    parser.add_argument("--rollin-beta", type=float, default=1.0, help="probability of using oracle for a full roll-in")
    parser.add_argument("--beta-step", type=float, default=0.00, help="per-epoch decrease of rollin-beta")
    parser.add_argument("--beta-burnin", type=int, default=0, help="number of epochs before we start to decrease beta")
    parser.add_argument("--self-teach-beta", default=1.0, help="")

    # loss
    parser.add_argument("--loss", choices=['multiset'], default='multiset')
    parser.add_argument("--self-teach-beta-step", type=float, default=0.05)

    # samplers & rollin
    parser.add_argument("--rollin", choices=["mixed"], default='mixed', help='learned rollin via --rollin-beta 0.0')
    parser.add_argument("--rollin-mix-type", choices=["trajectory", "state"], default="state", help="mix rollin at the trajectory or state level")
    parser.add_argument("--training-sampler", choices=["greedy", "stochastic", "policy_correct_greedy", "policy_correct_stochastic"], default="policy_correct_stochastic")
    parser.add_argument("--eval-sampler", choices=["greedy", "stochastic"], default="stochastic")

    # gpu
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=1234, help="seed")

    parser.add_argument("--no-visdom", action='store_true', help="don't use visdom when True")
