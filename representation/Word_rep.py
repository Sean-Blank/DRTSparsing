from random import random

import torch
import torch.nn as nn


class WordRep(nn.Module):
    def __init__(self, args, data_tem):
        super(WordRep, self).__init__()

        self.word_size = data_tem.word_vocab.size()
        self.char_size = data_tem.char_vocab.size()
        self.extra_vl_size = data_tem.extra_vocab.size()
        self.pre_train = data_tem.pre_train

        self.args = args
        self.word_embeds = nn.Embedding(self.word_size, args.word_dim)
        info_dim = args.word_dim
        if args.use_char:
            self.char_embeds = nn.Embedding(self.char_size, args.char_dim)
            self.lstm = nn.LSTM(args.char_dim, args.char_hidden_dim, num_layers=args.char_n_layer, bidirectional=True)
            info_dim += args.char_hidden_dim * 2 * 3
        if args.pretrain_path:
            self.pretrain_embeds = nn.Embedding(self.pre_train.size(), args.pretrain_dim)
            self.pretrain_embeds.weight = nn.Parameter(torch.FloatTensor(self.pre_train.vectors()), False)
            info_dim += args.pretrain_dim
        if args.extra_dim:
            dims = [args.extra_dim]

            extra_vl_size = [self.extra_vl_size]
            assert len(dims) == len(extra_vl_size)
            assert len(dims) <= 5, "5 extra embeds at most"
            if len(dims) >= 1:
                self.extra_embeds1 = nn.Embedding(extra_vl_size[0], int(dims[0]))
                info_dim += int(dims[0])
            if len(dims) >= 2:
                self.extra_embeds2 = nn.Embedding(extra_vl_size[1], int(dims[1]))
                info_dim += int(dims[1])
            if len(dims) >= 3:
                self.extra_embeds3 = nn.Embedding(extra_vl_size[2], int(dims[2]))
                info_dim += int(dims[2])
            if len(dims) >= 4:
                self.extra_embeds4 = nn.Embedding(extra_vl_size[3], int(dims[3]))
                info_dim += int(dims[3])
            if len(dims) >= 5:
                self.extra_embeds5 = nn.Embedding(extra_vl_size[4], int(dims[4]))
                info_dim += int(dims[4])

        self.info2input = nn.Linear(info_dim, args.input_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, instance, single_dict, train):
        word_sequence = []
        for i, widx in enumerate(instance[0], 0):
            if train and (widx in single_dict) and random() < self.args.single:
                word_sequence.append(instance[3][i])
            else:
                word_sequence.append(widx)

        word_t = torch.LongTensor(word_sequence).to(self.args.device)

        word_t = self.word_embeds(word_t)

        if train:
            word_t = self.dropout(word_t)

        if self.args.use_char:
            char_ts = []
            for char_instance in instance[1]:
                char_t = torch.LongTensor(char_instance).to(self.args.device)

                char_t = self.char_embeds(char_t)
                char_hidden_t = self.initcharhidden()
                char_t, _ = self.lstm(char_t.unsqueeze(1), char_hidden_t)
                char_t_avg = torch.sum(char_t, 0) / char_t.size(0)
                char_t_max = torch.max(char_t, 0)[0]
                char_t_min = torch.min(char_t, 0)[0]
                char_t_per_word = torch.cat((char_t_avg, char_t_max, char_t_min), 1)
                if train:
                    char_t_per_word = self.dropout(char_t_per_word)
                char_ts.append(char_t_per_word)
            char_t = torch.cat(char_ts, 0)
            word_t = torch.cat((word_t, char_t), 1)
        # print word_t, word_t.size()
        if self.args.pretrain_path:
            pretrain_t = torch.LongTensor(instance[2]).to(self.args.device)

            pretrain_t = self.pretrain_embeds(pretrain_t)
            word_t = torch.cat((word_t, pretrain_t), 1)
        # print word_t, word_t.size()
        if self.args.extra_dim:
            """
			for i, extra_embeds in enumerate(self.extra_embeds):
				extra_t = torch.LongTensor(instance[4+i])
				if self.args.gpu:
					extra_t = extra_t.cuda()
				extra_t = extra_embeds(extra_t)
				if not test:
					extra_t = self.dropout(extra_t)
				word_t = torch.cat((word_t, extra_t), 1)
			"""
            if len(instance) - 4 >= 1:
                extra_t = torch.LongTensor(instance[4 + 0]).to(self.args.device)

                extra_t = self.extra_embeds1(extra_t)
                word_t = torch.cat((word_t, extra_t), 1)
            if len(instance) - 4 >= 2:
                extra_t = torch.LongTensor(instance[4 + 1]).to(self.args.device)

                extra_t = self.extra_embeds1(extra_t)
                word_t = torch.cat((word_t, extra_t), 1)
            if len(instance) - 4 >= 3:
                extra_t = torch.LongTensor(instance[4 + 2]).to(self.args.device)

                extra_t = self.extra_embeds1(extra_t)
                word_t = torch.cat((word_t, extra_t), 1)
            if len(instance) - 4 >= 4:
                extra_t = torch.LongTensor(instance[4 + 3]).to(self.args.device)

                extra_t = self.extra_embeds1(extra_t)
                word_t = torch.cat((word_t, extra_t), 1)
            if len(instance) - 4 >= 5:
                extra_t = torch.LongTensor(instance[4 + 4]).to(self.args.device)

                extra_t = self.extra_embeds1(extra_t)
                word_t = torch.cat((word_t, extra_t), 1)

        word_t = self.tanh(self.info2input(word_t))

        return word_t

    def initcharhidden(self):
        result = (
            torch.zeros(2 * self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True).to(
                self.args.device),
            torch.zeros(2 * self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True).to(
                self.args.device))
        return result
