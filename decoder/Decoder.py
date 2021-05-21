from modules.Graph import *
from constraints.Constraints import *

class Decoder(nn.Module):
    def __init__(self, args, action_vocab):
        super(Decoder, self).__init__()
        self.action_size = action_vocab.size()
        self.args = args

        self.dropout = nn.Dropout(self.args.dropout)
        self.embeds = nn.Embedding(self.action_size, self.args.action_dim)

        self.struct2rel = nn.Linear(self.args.action_hidden_dim, self.args.action_dim)
        self.rel2var = nn.Linear(self.args.action_hidden_dim, self.args.action_dim)

        self.lstm = nn.LSTM(self.args.action_dim, self.args.action_hidden_dim, num_layers=self.args.action_layer_num)

        self.feat = nn.Linear(self.args.action_hidden_dim * 2 + self.args.action_dim, self.args.action_feature_dim)
        self.feat_tanh = nn.Tanh()

        self.out = nn.Linear(self.args.action_feature_dim, self.action_size)

        self.copy_head = nn.Linear(self.args.action_hidden_dim, self.args.action_hidden_dim, bias=False)
        self.copy = nn.Linear(self.args.action_hidden_dim, self.args.action_dim)

        self.word_head = nn.Linear(self.args.action_hidden_dim, self.args.action_hidden_dim, bias=False)
        self.sent_head = nn.Linear(self.args.action_hidden_dim, self.args.action_hidden_dim, bias=False)
        self.pointer_head = nn.Linear(self.args.action_hidden_dim, self.args.action_hidden_dim, bias=False)

        self.criterion = nn.NLLLoss()

        self.actn_v = action_vocab

        if self.args.gnn_rel == 1:
            self.graph_decoder_rel = GraphNet(self.args, 'dec_step2')

        self.cstn1, self.cstn2, self.cstn3 = get_constraints(self.args, self.actn_v)

    def forward(self, inputs, hidden, word_rep, sent_rep, pointer, copy_rep, train, state, opt):
        if opt == 1:
            return self.forward_1(inputs, hidden, word_rep, sent_rep, pointer, train, state)
        elif opt == 2:
            return self.forward_2(inputs, hidden, word_rep, sent_rep, copy_rep, train, state)
        elif opt == 3:
            return self.forward_3(inputs, hidden, word_rep, sent_rep, train, state)
        else:
            assert False, "unrecognized option"

    def forward_1(self, drs_gold, hidden, word_rep, sent_rep, pointer, train, state):
        if self.args.beam_size == 0:
            inputs = torch.LongTensor(drs_gold[:-1]).to(self.args.device)

            actions = self.embeds(inputs).unsqueeze(1)
            if train:
                actions = self.dropout(actions)
                self.lstm.dropout = self.args.dropout
            else:
                self.lstm.dropout = 0

            output, hidden = self.lstm(actions, hidden)

            assert len(pointer) == len(drs_gold)

            p = []  # the pointer of drs node, len(p) equals the number of DRS node
            drs_output = []  # the index of drs nodes in the input sequence
            for idrs, drs in enumerate(drs_gold[1:]):
                if self.actn_v.totok(drs) == "DRS(":
                    p.append(pointer[idrs])
                    drs_output.append(output[idrs].unsqueeze(0))
            drs_output = torch.cat(drs_output, 0)

            # word-level attention
            w_attn_scores = torch.bmm(self.word_head(output).transpose(0, 1), word_rep.transpose(1, 2))[0]
            w_attn_weights = F.softmax(w_attn_scores, 1)
            w_attn_hiddens = torch.bmm(w_attn_weights.unsqueeze(0), word_rep)[0]

            # sent-level attention
            s_attn_scores = torch.bmm(self.sent_head(output).transpose(0, 1), sent_rep.transpose(1, 2))[0]
            s_attn_weights = F.softmax(s_attn_scores, 1)
            s_attn_hiddens = torch.bmm(s_attn_weights.unsqueeze(0), sent_rep)[0]

            feat_hiddens = self.feat_tanh(
                self.feat(torch.cat((w_attn_hiddens, s_attn_hiddens, actions.view(output.size(0), -1)), 1)))
            global_scores = self.out(feat_hiddens)

            log_softmax_output = F.log_softmax(global_scores, 1)

            action_gold = torch.LongTensor(drs_gold[1:]).to(self.args.device)

            loss = self.criterion(log_softmax_output, action_gold)

            _, pre1 = torch.max(log_softmax_output, 1)

            acc1 = torch.sum(action_gold == pre1).data.tolist() / pre1.size()[0]

            # pointer attention
            p_attn_scores = torch.bmm(self.pointer_head(drs_output).transpose(0, 1), sent_rep.transpose(1, 2))[0]
            p_log_softmax = F.log_softmax(p_attn_scores, 1)
            pointer_gold = torch.LongTensor(p).to(self.args.device)

            loss_p = self.criterion(p_log_softmax, pointer_gold)

            _, prep = torch.max(p_log_softmax, 1)

            accp = torch.sum(pointer_gold == prep).data.tolist() / prep.size()[0]

            return loss, loss_p, output, hidden, acc1, accp
        else:
            self.lstm.dropout = 0
            tokens = []
            pointers = []
            hidden_rep = []
            pre_scores = []
            inputs = torch.LongTensor([drs_gold]).to(self.args.device)

            hidden_t = (
            hidden[0].view(self.args.action_layer_num, 1, -1), hidden[1].view(self.args.action_layer_num, 1, -1))
            action_t = self.embeds(inputs).unsqueeze(1)

            while True:
                output, hidden_t = self.lstm(action_t, hidden_t)
                hidden_rep.append(output)

                w_attn_scores_t = torch.bmm(self.word_head(output).transpose(0, 1), word_rep.transpose(1, 2))[0]
                w_attn_weights_t = F.softmax(w_attn_scores_t, 1)
                w_attn_hiddens_t = torch.bmm(w_attn_weights_t.unsqueeze(0), word_rep)[0]

                s_attn_scores_t = torch.bmm(self.sent_head(output).transpose(0, 1), sent_rep.transpose(1, 2))[0]
                s_attn_weights_t = F.softmax(s_attn_scores_t, 1)
                s_attn_hiddens_t = torch.bmm(s_attn_weights_t.unsqueeze(0), sent_rep)[0]

                feat_hiddens_t = self.feat_tanh(
                    self.feat(torch.cat((w_attn_hiddens_t, s_attn_hiddens_t, action_t.view(output.size(0), -1)), 1)))
                global_scores_t = self.out(feat_hiddens_t)

                pre_scores.append(global_scores_t)

                score_t = global_scores_t
                if self.args.const:
                    constraint = self.cstn1.get_step_mask(state)
                    constraint_t = torch.FloatTensor(constraint).unsqueeze(0).to(self.args.device)
                    score_t = global_scores_t + (constraint_t - 1.0) * 1e10

                _, input_t = torch.max(score_t, 1)
                idx = input_t.view(-1).data.tolist()[0]
                tokens.append(idx)

                self.cstn1.update(idx, state)

                if self.args.const:
                    if self.cstn1.isterminal(state):
                        break
                else:
                    if self.cstn1.bracket_completed(state) or len(tokens) > self.args.max_struct_l:
                        break
                action_t = self.embeds(input_t).view(1, 1, -1)

                # pointer
                if self.actn_v.totok(idx) == "DRS(":
                    p_attn_scores_t = torch.bmm(self.pointer_head(output).transpose(0, 1), sent_rep.transpose(1, 2))[
                        0]
                    _, p_t = torch.max(p_attn_scores_t, 1)
                    p_idx = p_t.view(-1).data.tolist()[0]
                    pointers.append(p_idx)

            return tokens, pointers, hidden_rep[0:], hidden_t

    def forward_2(self, rels_gold, hidden, word_rep, sent_rep, copy_rep, train, state):
        if self.args.beam_size == 0:
            outputs = []
            loss = []
            acc2 = []
            for struct, rels, p in rels_gold:

                inst = [self.struct2rel(struct).view(1, 1, -1)]
                for rel in rels[:-1]:  # rel( rel( rel( )
                    # assert type(rel) != type(None)
                    if isinstance(rel, str):
                        assert p != -1
                        inst.append(self.copy(copy_rep[p][int(rel[1:-1])].view(1, 1, -1)))
                    else:
                        rel = torch.LongTensor([rel]).to(self.args.device)
                        inst.append(self.embeds(rel).unsqueeze(0))
                actions = torch.cat(inst, 0)
                if train:
                    self.lstm.dropout = self.args.dropout
                    actions = self.dropout(actions)
                else:
                    self.lstm.dropout = 0

                output, hidden = self.lstm(actions, hidden)
                if self.args.gnn_rel == 1:
                    x = torch.cat((struct, output.squeeze(1)), 0)
                    edges = construct_graph_dec_rel(x, undirected=True)
                    edges = torch.LongTensor(edges).to(self.args.device)
                    x = self.graph_decoder_rel(x, edges, train)
                    x = x.narrow(0, 1, output.shape[0]).unsqueeze(1)
                    outputs.append(x)
                else:
                    outputs.append(output)

                copy_scores = None
                if p != -1:
                    copy_scores = torch.bmm(self.copy_head(output).transpose(0, 1),
                                            copy_rep[p].transpose(0, 1).unsqueeze(0)).view(output.size(0), -1)

                w_attn_scores = torch.bmm(self.word_head(output).transpose(0, 1), word_rep.transpose(1, 2))[0]
                w_attn_weights = F.softmax(w_attn_scores, 1)
                w_attn_hiddens = torch.bmm(w_attn_weights.unsqueeze(0), word_rep)[0]

                s_attn_scores = torch.bmm(self.sent_head(output).transpose(0, 1), sent_rep.transpose(1, 2))[0]
                s_attn_weights = F.softmax(s_attn_scores, 1)
                s_attn_hiddens = torch.bmm(s_attn_weights.unsqueeze(0), sent_rep)[0]

                feat_hiddens = self.feat_tanh(
                    self.feat(torch.cat((w_attn_hiddens, s_attn_hiddens, actions.view(output.size(0), -1)), 1)))
                global_scores = self.out(feat_hiddens)

                total_score = global_scores
                if p != -1:
                    total_score = torch.cat((total_score, copy_scores), 1)

                log_softmax_output = F.log_softmax(total_score, 1)

                actions_gold = []
                for i in range(len(rels)):
                    if type(rels[i]) == type('shabi'):
                        actions_gold.append(int(rels[i][1:-1]) + self.action_size)
                    else:
                        actions_gold.append(rels[i])

                actions_gold = torch.LongTensor(actions_gold).to(self.args.device)

                _, pre2 = torch.max(log_softmax_output, 1)

                acc2.append(torch.sum(actions_gold == pre2).data.tolist() / pre2.size()[0])

                loss.append(self.criterion(log_softmax_output, actions_gold).view(1, -1))

            acc2 = sum(acc2) / len(acc2)

            loss = torch.sum(torch.cat(loss, 0)) / len(loss)

            return loss, torch.cat(outputs, 0), hidden, acc2
        else:
            self.lstm.dropout = 0.0
            tokens = []
            hidden_rep = []
            input, p = rels_gold
            action_t = self.struct2rel(input).view(1, 1, -1)
            while True:
                output, hidden = self.lstm(action_t, hidden)
                # print dev_61
                hidden_rep.append(output)
                copy_scores_t = None
                if p != -1:
                    copy_scores_t = torch.bmm(self.copy_head(output).transpose(0, 1),
                                              copy_rep[p].transpose(0, 1).unsqueeze(0)).view(output.size(0), -1)

                # copy_scores_t = torch.bmm(torch.bmm(dev_61.transpose(0,1), self.copy_matrix), encoder_rep_t.transpose(0,1).unsqueeze(0)).view(dev_61.size(0), -1)
                # print copy_scores_t
                w_attn_scores_t = torch.bmm(self.word_head(output).transpose(0, 1), word_rep.transpose(1, 2))[0]
                w_attn_weights_t = F.softmax(w_attn_scores_t, 1)
                w_attn_hiddens_t = torch.bmm(w_attn_weights_t.unsqueeze(0), word_rep)[0]

                s_attn_scores_t = torch.bmm(self.sent_head(output).transpose(0, 1), sent_rep.transpose(1, 2))[0]
                s_attn_weights_t = F.softmax(s_attn_scores_t, 1)
                s_attn_hiddens_t = torch.bmm(s_attn_weights_t.unsqueeze(0), sent_rep)[0]

                feat_hiddens_t = self.feat_tanh(
                    self.feat(torch.cat((w_attn_hiddens_t, s_attn_hiddens_t, action_t.view(output.size(0), -1)), 1)))
                global_scores_t = self.out(feat_hiddens_t)

                total_score = global_scores_t
                if p != -1:
                    total_score = torch.cat((global_scores_t, copy_scores_t), 1)

                if self.args.const:
                    constraint = self.cstn2.get_step_mask(state)
                    constraint_t = torch.FloatTensor(constraint).unsqueeze(0).to(self.args.device)
                    total_score = total_score + (constraint_t - 1) * 1e10

                # print total_score
                # print total_score.view(-1).data.tolist()
                _, input_t = torch.max(total_score, 1)

                idx = input_t.view(-1).data.tolist()[0]
                tokens.append(idx)
                self.cstn2.update(idx, state)

                if self.args.const:
                    if self.cstn2.isterminal(state):
                        break
                else:
                    if self.cstn2.isterminal(state):
                        break
                    if len(tokens) > (self.args.rel_l + self.args.d_rel_l) * 2:
                        if self.args.soft_const:
                            tokens[-1] = self.actn_v.toidx(")")
                        else:
                            tokens = tokens[0:-1]  # last is not closed bracketd
                        break

                if idx >= self.action_size:
                    action_t = self.copy(copy_rep[p][idx - self.action_size].view(1, 1, -1))
                else:
                    action_t = self.embeds(input_t).view(1, 1, -1)

            if self.args.gnn_rel == 1:
                # hidden_rep[0] = input.unsqueeze(1)
                hidden_rep[0] = input
                x = torch.cat(hidden_rep, 0).squeeze(1)
                edges = construct_graph_dec_rel(x, undirected=True)
                if len(edges) != 0:
                    edges = torch.LongTensor(edges).to(self.args.device)
                    x = self.graph_decoder_rel(x, edges, train)
                    hidden_rep = x.unsqueeze(1).split(1, 0)

            return tokens, hidden_rep[1:], hidden, [state.rel_g, state.d_rel_g]

    def forward_3(self, vars_gold, hidden, word_rep, sent_rep, train, state):
        if self.args.beam_size == 0:
            actions = []
            actions_gold = []
            for rel, var in vars_gold:
                actions.append(self.rel2var(rel).view(1, 1, -1))
                actions_gold += var
                var = torch.LongTensor(var[:-1]).to(self.args.device)

                actions.append(self.embeds(var).unsqueeze(1))

            # print [x.size() for x in List]
            actions = torch.cat(actions, 0)
            # print action_t.size()
            if train:
                self.lstm.dropout = self.args.dropout
                actions = self.dropout(actions)
            else:
                self.lstm.dropout = 0

            output, hidden = self.lstm(actions, hidden)

            # word-level attention
            w_attn_scores = torch.bmm(self.word_head(output).transpose(0, 1), word_rep.transpose(1, 2))[0]
            w_attn_weights = F.softmax(w_attn_scores, 1)
            w_attn_hiddens = torch.bmm(w_attn_weights.unsqueeze(0), word_rep)[0]

            # sent-level attention
            s_attn_scores = torch.bmm(self.sent_head(output).transpose(0, 1), sent_rep.transpose(1, 2))[0]
            s_attn_weights = F.softmax(s_attn_scores, 1)
            s_attn_hiddens = torch.bmm(s_attn_weights.unsqueeze(0), sent_rep)[0]

            feat_hiddens = self.feat_tanh(
                self.feat(torch.cat((w_attn_hiddens, s_attn_hiddens, actions.view(output.size(0), -1)), 1)))
            global_scores = self.out(feat_hiddens)

            log_softmax_output = F.log_softmax(global_scores, 1)

            actions_gold = torch.LongTensor(actions_gold).to(self.args.device)

            loss = self.criterion(log_softmax_output, actions_gold)

            _, pre3 = torch.max(log_softmax_output, 1)
            acc3 = float(torch.sum(actions_gold == pre3).cpu().numpy().item()) / len(pre3)

            return loss, output, hidden, acc3
        else:
            self.lstm.dropout = 0.0
            tokens = []
            action_t = self.rel2var(vars_gold).view(1, 1, -1)
            while True:
                output, hidden = self.lstm(action_t, hidden)

                w_attn_scores_t = torch.bmm(self.word_head(output).transpose(0, 1), word_rep.transpose(1, 2))[0]
                w_attn_weights_t = F.softmax(w_attn_scores_t, 1)
                w_attn_hiddens_t = torch.bmm(w_attn_weights_t.unsqueeze(0), word_rep)[0]

                s_attn_scores_t = torch.bmm(self.sent_head(output).transpose(0, 1), sent_rep.transpose(1, 2))[0]
                s_attn_weights_t = F.softmax(s_attn_scores_t, 1)
                s_attn_hiddens_t = torch.bmm(s_attn_weights_t.unsqueeze(0), sent_rep)[0]

                feat_hiddens_t = self.feat_tanh(
                    self.feat(torch.cat((w_attn_hiddens_t, s_attn_hiddens_t, action_t.view(output.size(0), -1)), 1)))
                global_scores_t = self.out(feat_hiddens_t)
                # print "global_scores_t", global_scores_t

                score_t = global_scores_t
                if self.args.const:
                    constraint = self.cstn3.get_step_mask(state)
                    constraint_t = torch.FloatTensor(constraint).unsqueeze(0).to(self.args.device)
                    score_t = global_scores_t + (constraint_t - 1.0) * 1e10

                # print score
                _, input_t = torch.max(score_t, 1)
                idx = input_t.view(-1).data.tolist()[0]

                tokens.append(idx)

                self.cstn3.update(idx, state)

                if self.args.const:
                    if self.cstn3.isterminal(state):
                        break
                else:
                    if self.cstn3.isterminal(state):
                        break
                    if len(tokens) >= 3:
                        if self.args.soft_const:
                            tokens[-1] = self.actn_v.toidx(")")
                        break
                action_t = self.embeds(input_t).view(1, 1, -1)

            return tokens, None, hidden, [state.x, state.e, state.s, state.t]
