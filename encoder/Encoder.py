from modules.Graph import *


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args

        self.lstm = nn.LSTM(args.input_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_layer_num, bidirectional=True)
        self.sent_lstm = nn.LSTM(args.bilstm_hidden_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_layer_num, bidirectional=True)
        self.word2sent = nn.Linear(args.bilstm_hidden_dim*2, args.bilstm_hidden_dim)
        if self.args.gnn_word == 1:
            self.graph_encoder_word = GraphNet(self.args, 'word')
        if self.args.gnn_sent == 1:
            self.graph_encoder_sent = GraphNet(self.args, 'sent')

    def forward(self, inst, combs, seps, word_graph, train):
        hidden = self.init_hidden()
        if train:
            self.lstm.dropout = self.args.dropout
            self.sent_lstm.dropout = self.args.dropout
        else:
            self.lstm.dropout = 0
            self.sent_lstm.dropout = 0

        output, hidden_word = self.lstm(inst.unsqueeze(1), hidden)
        if self.args.gnn_word == 1:
            word_graph = torch.LongTensor(word_graph).to(self.args.device)
            output = self.graph_encoder_word(output.squeeze(1), word_graph, train)
            output = output.unsqueeze(1)

        assert len(seps) - 1 == len(combs)

        copy_rep_s = []
        for i in range(len(seps)-1):
            s = seps[i]
            e = seps[i+1]
            sent = output[s+1:e]
            comb = combs[i]

            copy_rep = []
            for j in range(len(comb)):
                copy_rep.append([])
                for idx in comb[j]:
                    copy_rep[-1].append(sent[idx])
                copy_rep[-1] = (torch.sum(torch.cat(copy_rep[-1]), 0)/(len(comb[j]))).unsqueeze(0)
            copy_rep = torch.cat(copy_rep, 0)
            copy_rep_s.append(copy_rep)

        sent_rep = []
        for i in range(len(seps)-1):
            s = seps[i]
            e = seps[i+1]
            sent_rep.append(torch.cat([output[e].view(2, -1)[0], output[s].view(2, -1)[1]]).view(1, -1))
        sent_rep = torch.cat(sent_rep, 0).unsqueeze(1)
        # 这里可以用hidden_word初始化？
        hidden_sent = self.init_hidden()
        sent_rep, hidden_sent = self.sent_lstm(self.word2sent(sent_rep), hidden_sent)
        if self.args.gnn_sent == 1:
            train_enc_sent_graph = construct_graph_allsents(sent_rep)
            if len(train_enc_sent_graph) != 0:
                train_enc_sent_graph = torch.LongTensor(train_enc_sent_graph).to(self.args.device)
                sent_rep = self.graph_encoder_sent(sent_rep.squeeze(1), train_enc_sent_graph, train)
                sent_rep = sent_rep.unsqueeze(1)

        return output.transpose(0, 1), sent_rep.transpose(0, 1), copy_rep_s, hidden_sent

    def init_hidden(self):

        result = (torch.zeros(2*self.args.bilstm_layer_num, 1, self.args.bilstm_hidden_dim, requires_grad=True).to(self.args.device),
                  torch.zeros(2*self.args.bilstm_layer_num, 1, self.args.bilstm_hidden_dim, requires_grad=True).to(self.args.device))
        return result
