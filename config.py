import argparse
import json

def parse_config(parser):
    #
    parser.add_argument("--random_seed", type=int, default=19940206)
    parser.add_argument("--check_steps", default=1000)
    parser.add_argument("--validate_steps", default=10000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_token", default=False)

    #
    parser.add_argument("--vocab_min_freq", type=int, default=5)
    # path
    parser.add_argument("--train_data", default='./data/dev')
    parser.add_argument("--dev_data", default='./data/dev')
    parser.add_argument("--test_data", default='./data/test')
    parser.add_argument("--token_vocab", default='./data/token_vocab')
    parser.add_argument("--token_char_vocab", default='./data/token_char_vocab')
    parser.add_argument("--concept_vocab", default='./data/concept_vocab')
    parser.add_argument("--concept_char_vocab", default='./data/concept_char_vocab')
    parser.add_argument("--relation_vocab", default='./data/relation_vocab')
    parser.add_argument("--ckpt", default='./ckpt')
    parser.add_argument("--log_dir", default='./ckpt')
    parser.add_argument("--suffix", default='amr')

    # concept/token encoders
    parser.add_argument('--word_char_dim', default=32)
    parser.add_argument('--word_dim', default=300)
    parser.add_argument('--concept_char_dim', default=32)
    parser.add_argument('--concept_dim', default=300)

    # char-cnn
    parser.add_argument('--cnn_filters', default=[3, 256], nargs='+')
    parser.add_argument('--char2word_dim', default=128)
    parser.add_argument('--char2concept_dim', default=128)

    # sent encoder
    parser.add_argument('--rnn_hidden_size', default=256)
    parser.add_argument('--rnn_num_layers', default=2)

    # core architecture
    parser.add_argument('--ff_embed_dim', default=1024)
    parser.add_argument('--ffnn_depth', default=2)
    parser.add_argument('--gnn_layers', default=4)
    parser.add_argument('--coref_depth', default=3)
    parser.add_argument("--embed_dim", default=512)
    parser.add_argument("--dropout", default=0.2)
    parser.add_argument("--learning_rate", default=1e-3)
    parser.add_argument("--bert_learning_rate", default=1e-4)
    parser.add_argument("--num_epochs", default=10)
    parser.add_argument("--grad_accum_steps", default=5)
    parser.add_argument("--warmup_proportion", default=0.0)
    parser.add_argument("--antecedent_max_num", default=50)
    parser.add_argument("--use_speaker", default=False)
    return parser


def save_config(args, out_path):
    args_dict = vars(args)
    with open(out_path, 'w') as fp:
        json.dump(args_dict, fp)


def load_config(in_path):
    with open(in_path, 'r') as fp:
        args_dict = json.load(fp)
        return args_dict