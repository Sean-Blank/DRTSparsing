from representation.Word_rep import *
from encoder.Encoder import Encoder as Enc
from decoder.Decoder import Decoder as Dec
from constraints.Constraints import *
from modules.Graph import *
from modules.utils import *


class DRSparsing(nn.Module):
    def __init__(self, args, data_init):
        super(DRSparsing, self).__init__()
        # self.action_size = data_init.action_vocab.size()
        self.args = args
        self.encoder = Enc(self.args)
        self.decoder = Dec(self.args, data_init.action_vocab)
        self.input_rep = WordRep(args, data_init)
        if self.args.gnn_str == 1:
            self.graph_decoder_str = GraphNet(self.args, 'dec_step1')

    def forward(self, data, i, train):
        if self.args.beam_size == 0:
            inst = self.input_rep(data.instances[i], single_dict=data.single_dict, train=train)

            word_rep, sent_rep, copy_rep, hidden = self.encoder(inst, data.combs[i], data.seps[i],
                                                                data.enc_word_graph[i], train=train)

            # ************************************************decoder_step1********************************************
            hidden_step1 = (
                hidden[0].view(self.args.action_layer_num, 1, -1), hidden[1].view(self.args.action_layer_num, 1, -1))
            loss_1, loss_p, hidden_rep, hidden_step1, acc1, accp = self.decoder(data.actions[i][0], hidden_step1,
                                                                                word_rep, sent_rep,
                                                                                pointer=data.actions[i][3],
                                                                                copy_rep=None, train=train, state=None,
                                                                                opt=1)

            if self.args.gnn_str == 1:
                if len(data.dec_str_graph[i]) != 0:
                    graph_str = torch.LongTensor(data.dec_str_graph[i]).to(self.args.device)
                    hidden_rep = self.graph_decoder_str(hidden_rep.squeeze(1), graph_str, train)
                    hidden_rep = hidden_rep.unsqueeze(1)

            # ************************************************decoder_step2********************************************
            idx = 0
            hidden_step2 = (
                hidden[0].view(self.args.action_layer_num, 1, -1), hidden[1].view(self.args.action_layer_num, 1, -1))
            # hidden_step2 = hidden_step1
            train_action_step2 = []

            for j in range(len(data.actions[i][0])):  # <START> DRS( P1(
                tok = data.actions[i][0][j]
                if data.action_vocab.totok(tok) == "DRS(":
                    train_action_step2.append(
                        [hidden_rep[j], data.actions[i][1][idx], data.actions[i][4][idx][0]])

                    idx += 1
                elif data.action_vocab.totok(tok) == "SDRS(":
                    train_action_step2.append([hidden_rep[j], data.actions[i][1][idx], -1])
                    idx += 1

            assert idx == len(data.actions[i][1])
            loss_2, hidden_rep, hidden_step2, acc2 = self.decoder(train_action_step2, hidden_step2, word_rep, sent_rep,
                                                                  pointer=None, copy_rep=copy_rep, train=train,
                                                                  state=None,
                                                                  opt=2)

            # ************************************************decoder_step3********************************************
            flat_train_action = [0]  # <START>
            for l in data.actions[i][1]:
                flat_train_action += l

            idx = 0
            hidden_step3 = (
                hidden[0].view(self.args.action_layer_num, 1, -1), hidden[1].view(self.args.action_layer_num, 1, -1))
            # hidden_step3 = hidden_step2
            train_action_step3 = []

            for j in range(len(flat_train_action)):
                tok = flat_train_action[j]
                if (type(tok) == type('hehe') and tok[-1] == "(") or data.action_vocab.totok(tok)[-1] == "(":
                    train_action_step3.append([hidden_rep[j], data.actions[i][2][idx]])
                    idx += 1

            loss_3, hidden_rep, hidden_step3, acc3 = self.decoder(train_action_step3, hidden_step3, word_rep, sent_rep,
                                                                  pointer=None, copy_rep=None, train=train, state=None,
                                                                  opt=3)
            return loss_1, loss_2, loss_3, loss_p, acc1, acc2, acc3, accp
        else:
            state_step1 = StructConstraintsState()
            state_step2 = RelationConstraintsState()
            state_step3 = VariableConstraintsState()

            w = open(self.args.output_file, 'w', encoding="utf-8")
            for j, (instance, comb, sep, enc_word_graph, test_input) in enumerate(zip(data.instances, data.combs, data.seps, data.enc_word_graph, data.inputs)):
                print(j)

                inst = self.input_rep(instance, single_dict=None, train=False)
                word_rep, sent_rep, copy_rep, hidden = self.encoder(inst, comb, sep, enc_word_graph, train=False)
                # word            sent             word             sent

                # step 1
                hidden_step1 = (hidden[0].view(self.args.action_layer_num, 1, -1), hidden[1].view(self.args.action_layer_num, 1, -1))
                state_step1.reset()
                output_step1, pointers_step1, hidden_rep_step1, hidden_step1 = self.decoder(
                    data.action_vocab.toidx("<START>"), hidden_step1, word_rep, sent_rep, pointer=None,
                    copy_rep=None,
                    train=False, state=state_step1, opt=1)

                if self.args.gnn_str == 1:
                    if len(output_step1) != 0:
                        graph_str = construct_graph_evl_str(output_step1, undirected=True)
                        graph_str = torch.LongTensor(graph_str).to(self.args.device)
                        hidden_rep_step1 = self.graph_decoder_str(torch.cat(hidden_rep_step1, 0).squeeze(1), graph_str, train)
                        hidden_rep_step1 = hidden_rep_step1.unsqueeze(1)
                        hidden_rep_step1 = hidden_rep_step1[1:]

                # step 2
                output_step2 = []
                hidden_rep_step2 = []
                hidden_step2 = (
                    hidden[0].view(self.args.action_layer_num, 1, -1), hidden[1].view(self.args.action_layer_num, 1, -1))

                state_step2.reset()  # <s> </s>
                drs_idx = 0
                for k in range(len(output_step1)):  # DRS( P1(
                    act1 = output_step1[k]
                    act2 = None
                    if k + 1 < len(output_step1):
                        act2 = output_step1[k + 1]
                    if data.action_vocab.totok(act1) == "DRS(":
                        # print "DRS",test_pointers_step1[drs_idx]
                        state_step2.reset_length(copy_rep[pointers_step1[drs_idx]].size(0))
                        state_step2.reset_condition(act1, act2)
                        one_output_step2, one_hidden_rep_step2, hidden_step2, partial_state = self.decoder(
                            [hidden_rep_step1[k], pointers_step1[drs_idx]], hidden_step2,
                            word_rep,
                            sent_rep, pointer=None, copy_rep=copy_rep, train=False, state=state_step2,
                            opt=2)
                        output_step2.append(one_output_step2)
                        hidden_rep_step2.append(one_hidden_rep_step2)
                        # partial_state is to store how many relation it already has
                        state_step2.rel_g, state_step2.d_rel_g = partial_state
                        drs_idx += 1
                    if data.action_vocab.totok(act1) == "SDRS(":
                        # print "SDRS"
                        state_step2.reset_length(0)
                        state_step2.reset_condition(act1, act2)
                        one_output_step2, one_hidden_rep_step2, hidden_step2, partial_state = self.decoder(
                            [hidden_rep_step1[k], -1], hidden_step2, word_rep, sent_rep,
                            pointer=None,
                            copy_rep=copy_rep, train=False, state=state_step2, opt=2)
                        output_step2.append(one_output_step2)
                        hidden_rep_step2.append(one_hidden_rep_step2)
                        # partial_state is to store how many relation it already has
                        state_step2.rel_g, state_step2.d_rel_g = partial_state

                # print test_hidden_step2

                # print one_test_hidden_rep_step2
                # print test_hidden_step2
                # exit(1)
                # print test_output_step2
                # step 3
                k_scope = get_k_scope(output_step1, data.action_vocab)
                p_max = get_p_max(output_step1, data.action_vocab)

                test_output_step3 = []
                test_hidden_step3 = (
                    hidden[0].view(self.args.action_layer_num, 1, -1),
                    hidden[1].view(self.args.action_layer_num, 1, -1))
                state_step3.reset(p_max)
                k = 0
                sdrs_idx = 0
                for act1 in output_step1:
                    if data.action_vocab.totok(act1) in ["DRS(", "SDRS("]:
                        if data.action_vocab.totok(act1) == "SDRS(":
                            state_step3.reset_condition(act1, k_scope[sdrs_idx])
                            sdrs_idx += 1
                        else:
                            state_step3.reset_condition(act1)
                        for kk in range(len(output_step2[k]) - 1):  # rel( rel( )
                            act2 = output_step2[k][kk]

                            state_step3.reset_relation(act2)
                            # print test_hidden_rep_step2[k][kk]
                            # print test_hidden_step3
                            # print "========================="
                            one_test_output_step3, _, test_hidden_step3, partial_state = self.decoder(
                                hidden_rep_step2[k][kk],
                                test_hidden_step3,
                                word_rep,
                                sent_rep, pointer=None,
                                copy_rep=None, train=False,
                                state=state_step3, opt=3)
                            test_output_step3.append(one_test_output_step3)
                            # partial state is to store how many variable it already has
                            state_step3.x, state_step3.e, state_step3.s, state_step3.t = partial_state

                        k += 1


                test_output = []
                k = 0
                kk = 0
                drs_idx = 0
                for act1 in output_step1:
                    if data.action_vocab.totok(act1) == "DRS(":
                        assert drs_idx < len(pointers_step1)
                        test_output.append("DRS-" + str(pointers_step1[drs_idx]) + "(")
                        drs_idx += 1
                    else:
                        test_output.append(data.action_vocab.totok(act1))
                    if data.action_vocab.totok(act1) in ["DRS(", "SDRS("]:
                        for act2 in output_step2[k][:-1]:
                            if act2 >= data.action_vocab.size():
                                test_output.append("$" + str(act2 - data.action_vocab.size()) + "(")
                            else:
                                test_output.append(data.action_vocab.totok(act2))
                            for act3 in test_output_step3[kk]:
                                test_output.append(data.action_vocab.totok(act3))
                            kk += 1
                        k += 1
                w.write(out_tree(test_input[1], test_output) + "\n")
                w.flush()
                assert drs_idx == len(pointers_step1)
            w.close()
