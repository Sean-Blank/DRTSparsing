import argparse
import torch
print(torch.cuda.is_available())
from vocab.Data import *
from representation.Word_rep import *
from encoder.Encoder import Encoder as Enc
from decoder.Decoder import Decoder as Dec
from constraints.Constraints import *

from modules.DRSparsing import *
import math


def train(args, model, train_data, dev_data, test_data):

    model_optimizer = optimizer(args, model.parameters())
    # lr_schedular = optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[10, 100], gamma=0.2)
    lr_schedular = optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min', 0.5, patience=3, verbose=False, min_lr=1e-5)

    iteration = len(train_data.instances)
    check_iter = 0
    check_loss1, check_loss2, check_loss3, check_lossp = 0, 0, 0, 0
    check_acc1, check_acc2, check_acc3, check_accp = 0, 0, 0, 0

    epoch = -1
    while epoch < 150:
        model.train()
        model_optimizer.zero_grad()
        if iteration == len(train_data.instances):
            iteration = 0
            epoch += 1

        check_iter += 1

        loss_1, loss_2, loss_3, loss_p, acc1, acc2, acc3, accp = model(train_data, iteration, train=True)

        iteration += 1
        loss = loss_1 + loss_2 + loss_3 + loss_p
        # schedular.step(loss)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5)
        model_optimizer.step()
        lr = [group['lr'] for group in model_optimizer.param_groups]
        # ************************************************check_train_loss********************************************

        check_loss1 += loss_1.data.tolist()
        check_loss2 += loss_2.data.tolist()
        check_loss3 += loss_3.data.tolist()
        check_lossp += loss_p.data.tolist()
        check_acc1 += acc1
        check_acc2 += acc2
        check_acc3 += acc3
        check_accp += accp


        if check_iter % args.check_per_update == 0:

            check_loss1 = check_loss1 / args.check_per_update
            check_loss2 = check_loss2 / args.check_per_update
            check_loss3 = check_loss3 / args.check_per_update
            check_lossp = check_lossp / args.check_per_update

            check_acc1 = check_acc1 / args.check_per_update
            check_acc2 = check_acc2 / args.check_per_update
            check_acc3 = check_acc3 / args.check_per_update
            check_accp = check_accp / args.check_per_update

            check_loss = check_loss1 + check_lossp + check_loss2 + check_loss3
            check_acc = (check_acc1 + check_acc2 + check_acc3 + check_accp) / 4

            print('[Epoch %d]:  Train loss: %.4f   Train ppl: %.4f  Train acc:  %.4f  Lr: %f' % (epoch, check_loss, math.exp(check_loss), check_acc, lr[0]))

            check_loss1, check_loss2, check_loss3, check_lossp = 0, 0, 0, 0
            check_acc1, check_acc2, check_acc3, check_accp = 0, 0, 0, 0

        # ************************************************save and validate********************************************
        if check_iter % args.validate_per_update == 0:
            torch.save(model.state_dict(), args.model_path + "/model" + str(int(check_iter / args.validate_per_update)))

            print('******************************************************************************************')
            print('Evaluating checkpoint ' + str(int(check_iter / args.validate_per_update)) + ':')

            dev_l, dev_l1, dev_l2, dev_l3, dev_lp, dev_acc, dev_acc1, dev_acc2, dev_acc3, dev_accp = validate(dev_data, model)

            print('      Dev:  loss: %.4f   ppl: %.4f   acc: %.4f' % (dev_l, math.exp(dev_l), dev_acc))

            lr_schedular.step(dev_l)

            test_l, test_l1, test_l2, test_l3, test_lp, test_acc, test_acc1, test_acc2, test_acc3, test_accp = validate(test_data, model)

            print('     Test:  loss: %.4f   ppl: %.4f   acc: %.4f' % (test_l, math.exp(test_l), test_acc))

            print('******************************************************************************************')



def validate(data, model):
    loss1, loss2, loss3, lossp = 0, 0, 0, 0
    acc_1, acc_2, acc_3, acc_p = 0, 0, 0, 0
    train_num = len(data.instances)

    with torch.no_grad():
        model.eval()
        for iteration in range(train_num):

            loss_1, loss_2, loss_3, loss_p, acc1, acc2, acc3, accp = model(data, iteration, train=False)

            loss1 += loss_1.data.tolist()
            loss2 += loss_2.data.tolist()
            loss3 += loss_3.data.tolist()
            lossp += loss_p.data.tolist()
            acc_1 += acc1
            acc_2 += acc2
            acc_3 += acc3
            acc_p += accp

        loss1 = loss1 / train_num
        loss2 = loss2 / train_num
        loss3 = loss3 / train_num
        lossp = lossp / train_num

        loss = loss1 + loss2 + loss3 + lossp

        acc_1 = acc_1 / train_num
        acc_2 = acc_2 / train_num
        acc_3 = acc_3 / train_num
        acc_p = acc_p / train_num

        acc = (acc_1 + acc_2 + acc_3 + acc_p) / 4

    return loss, loss1, loss2, loss3, lossp, acc, acc_1, acc_2, acc_3, acc_p


def set_hypers(parser):
    parser.add_argument("--gnn_word", type=int, default=0)
    parser.add_argument("--gnn_sent", type=int, default=0)
    parser.add_argument("--gnn_str", type=int, default=1)
    parser.add_argument("--gnn_rel", type=int, default=0)
    parser.add_argument("--gnn_type", default='gat')
    parser.add_argument("--gnn_layer", type=int, default=2)
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

    parser.add_argument("--model_path", default='checkpoints/models')
    #
    parser.add_argument("--train_input", default='data/dev.sent.pre.oracle.in')
    parser.add_argument("--train_action", default='data/dev.sent.pre.oracle.out')
    parser.add_argument("--train_graph", default='data/dev.sent.graph')
    parser.add_argument("--dev_input", default='data/dev.sent.pre.oracle.in')
    parser.add_argument("--dev_action", default='data/dev.sent.pre.oracle.out')
    parser.add_argument("--dev_graph", default='data/dev.sent.graph')
    parser.add_argument("--test_input", default='data/test.sent.pre.oracle.in')
    parser.add_argument("--test_action", default='data/test.sent.pre.oracle.out')
    parser.add_argument("--test_graph", default='data/test.sent.graph')
    parser.add_argument("--pretrain_path", default='data/embeddings/test.vectors')
    parser.add_argument("--action_dict_path", default='data/dict')

    # parser.add_argument("--train_input", default='data/evl.doc.pre.oracle.in')
    # parser.add_argument("--train_action", default='data/evl.doc.pre.oracle.out')
    # parser.add_argument("--train_graph", default='data/evl.doc.graph')
    # parser.add_argument("--dev_input", default='data/evl.doc.pre.oracle.in')
    # parser.add_argument("--dev_action", default='data/evl.doc.pre.oracle.out')
    # parser.add_argument("--dev_graph", default='data/evl.doc.graph')
    # parser.add_argument("--test_input", default='data/evl.doc.pre.oracle.in')
    # parser.add_argument("--test_action", default='data/evl.doc.pre.oracle.out')
    # parser.add_argument("--test_graph", default='data/evl.doc.graph')
    # parser.add_argument("--pretrain_path", default='data/embeddings/test.vectors')
    # parser.add_argument("--action_dict_path", default='data/dict')

    # parser.add_argument("--train_input", default='data/evl.doc.oracle.in')
    # parser.add_argument("--train_action", default='data/evl.doc.pre.oracle.out')
    # parser.add_argument("--train_graph", default='data/evl.doc.graph')
    # parser.add_argument("--dev_input", default='data/evl.doc.pre.oracle.in')
    # parser.add_argument("--dev_action", default='data/evl.doc.pre.oracle.out')
    # parser.add_argument("--dev_graph", default='data/evl.doc.graph')
    # parser.add_argument("--test_input", default='data/evl.doc.pre.oracle.in')
    # parser.add_argument("--test_action", default='data/evl.doc.pre.oracle.out')
    # parser.add_argument("--test_graph", default='data/evl.doc.graph')
    # parser.add_argument("--pretrain_path", default='data/embeddings/test.vectors')
    # parser.add_argument("--action_dict_path", default='data/dict')

    parser.add_argument("--check_per_update", default=10000)
    parser.add_argument("--validate_per_update", default=50000)
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

    train_data = Data(args)
    dev_data = Data(args)
    test_data = Data(args)

    train_data.load_data('train')
    dev_data.load_data('dev')
    test_data.load_data('test')

    model = DRSparsing(args, train_data).to(args.device)

    train(args, model, train_data, dev_data, test_data)


