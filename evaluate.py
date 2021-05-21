import argparse
import torch

from modules.DRSparsing import *
from vocab.Data import *


    #return bleu, match_num

def set_hypers(parser):
    parser.add_argument("--gnn_word", type=int, default=0)
    parser.add_argument("--gnn_sent", type=int, default=0)
    parser.add_argument("--gnn_str", type=int, default=0)
    parser.add_argument("--gnn_rel", type=int, default=0)
    parser.add_argument("--gnn_type", default='gat')
    parser.add_argument("--gnn_layer", default=2)
    parser.add_argument("--heads", default=4)
    parser.add_argument("--out_heads", default=4)
    parser.add_argument("--dropout_gnn", type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument("--residual", default=False)

    parser.add_argument("--random_seed", type=int, default=123456789)
    parser.add_argument("--word_dim", type=int, default=300)
    parser.add_argument("--char_dim", type=int, default=20)
    parser.add_argument("--pretrain_dim", type=int, default=100)
    parser.add_argument("--extra_dim", type=int, default=100)
    parser.add_argument("--input_dim", type=int, default=100)
    parser.add_argument("--bilstm_hidden_dim", type=int, default=300)
    parser.add_argument("--bilstm_layer_num", type=int, default=2)
    parser.add_argument("--action_dim", type=int, default=300)
    parser.add_argument("--action_hidden_dim", type=int, default=600)
    parser.add_argument("--action_layer_num", type=int, default=2)
    parser.add_argument("--action_feature_dim", type=int, default=150)

    parser.add_argument("--use_char", action='store_true')
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--learning_rate_decay", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--single", type=float, default=0.1)

    parser.add_argument("--K_l", type=int, default=35)
    parser.add_argument("--E_l", type=int, default=55)
    parser.add_argument("--P_l", type=int, default=25)
    parser.add_argument("--S_l", type=int, default=55)
    parser.add_argument("--T_l", type=int, default=45)
    parser.add_argument("--X_l", type=int, default=170)
    parser.add_argument("--drs_l", type=int, default=60)
    parser.add_argument("--rel_l", type=int, default=125)
    parser.add_argument("--rel_g_l", type=int, default=575)
    parser.add_argument("--d_rel_l", type=int, default=30)
    parser.add_argument("--d_rel_g_l", type=int, default=35)

    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--beam_size", default=0)

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = set_hypers(parser)

    parser.add_argument("--model_path", default='checkpoints/models_word')
    #
    parser.add_argument("--train_input", default='data/train.sent.pre.oracle.in')
    parser.add_argument("--train_action", default='data/train.sent.pre.oracle.out')
    parser.add_argument("--train_graph", default='data/train.sent.graph')
    parser.add_argument("--dev_input", default='data/dev.sent.pre.oracle.in')
    parser.add_argument("--dev_action", default='data/dev.sent.pre.oracle.out')
    parser.add_argument("--dev_graph", default='data/dev.sent.graph')
    parser.add_argument("--test_input", default='data/test.sent.pre.oracle.in')
    parser.add_argument("--test_action", default='data/test.sent.pre.oracle.out')
    parser.add_argument("--test_graph", default='data/test.sent.graph')
    parser.add_argument("--pretrain_path", default='data/embeddings/sskip.100.vectors')
    parser.add_argument("--action_dict_path", default='data/dict')

    # parser.add_argument("--train_input", default='data_doc/evl.oracle.doc.in')
    # parser.add_argument("--train_action", default='data_doc/evl.oracle.doc.out')
    # parser.add_argument("--train_graph", default='data_doc/evl.graph')
    # parser.add_argument("--dev_input", default='data_doc/evl.oracle.doc.in')
    # parser.add_argument("--dev_action", default='data_doc/evl.oracle.doc.out')
    # parser.add_argument("--dev_graph", default='data_doc/evl.graph')
    # parser.add_argument("--test_input", default='data_doc/evl.oracle.doc.in')
    # parser.add_argument("--test_action", default='data_doc/evl.oracle.doc.out')
    # parser.add_argument("--test_graph", default='data_doc/evl.graph')
    # parser.add_argument("--pretrain_path", default='data_doc/embeddings/test.vectors')
    # parser.add_argument("--action_dict_path", default='data_doc/dict')

    parser.add_argument("--check_per_update", default=1000)
    parser.add_argument("--validate_per_update", default=10000)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    print("GPU available: %s      CuDNN: %s" % (torch.cuda.is_available(), torch.backends.cudnn.enabled))
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available() and args.gpu >= 0:
        print("Using GPU To Train...    GPU ID: ", args.gpu)
        args.device = torch.device('cuda', args.gpu)
        torch.cuda.manual_seed(args.random_seed)
    else:
        args.device = torch.device('cpu')
        print("Using CPU To Train... ")

    args.beam_size = 1
    args.const = True
    args.soft_const = False
    args.model_path = 'checkpoints/models'

    dev_data = Data(args)
    test_data = Data(args)

    dev_data.load_data('dev')
    test_data.load_data('test')
    # model_idx = 30
    for model_idx in [30, 31, 32]:
        model_name = args.model_path + "/model" + str(model_idx)
        model = DRSparsing(args, dev_data).to(args.device)
        model.args.gpu = 0
        model.load_state_dict(torch.load(model_name, map_location={'cuda:1': 'cuda:0'}))

        model.eval()
        model.args.output_file = args.model_path + '/dev_' + str(model_idx)
        model(dev_data, '', train=False)
        model.args.output_file = args.model_path + "/test_" + str(model_idx)
        model(test_data, '', train=False)

