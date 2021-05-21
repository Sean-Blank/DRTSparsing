from vocab.Vocabulary import *
from modules.utils import *
from modules.Graph import *
from vocab.PretrainedEmb import PretrainedEmb


class Data:
    def __init__(self, args):
        self.args = args

        # 词表
        self.word_vocab = None
        self.char_vocab = None
        self.action_vocab = None
        self.extra_vocab = None
        # pre_train
        self.pre_train = None
        # train_data
        self.inputs = []
        self.seps = []
        self.combs = []
        self.instances = None
        self.single_dict = None
        self.word_dict = None

        self.enc_word_graph = []
        self.enc_sent_graph = []
        self.dec_str_graph = []
        self.dec_rel_graph = []

        self.trees = []
        self.actions = []
        self.create_vocab()


    def create_vocab(self):
        self.word_vocab = Vocabulary()
        self.char_vocab = Vocabulary()
        self.action_vocab = Vocabulary(UNK=False)
        self.extra_vocab = Vocabulary()
        self.pre_train = PretrainedEmb(self.args.pretrain_path)

        self.action_vocab.toidx("<START>")
        self.action_vocab.toidx("<END>")
        for i in range(self.args.X_l):
            self.action_vocab.toidx("X" + str(i + 1))
        for i in range(self.args.E_l):
            self.action_vocab.toidx("E" + str(i + 1))
        for i in range(self.args.S_l):
            self.action_vocab.toidx("S" + str(i + 1))
        for i in range(self.args.T_l):
            self.action_vocab.toidx("T" + str(i + 1))
        for i in range(self.args.P_l):
            self.action_vocab.toidx("P" + str(i + 1))
        for i in range(self.args.K_l):
            self.action_vocab.toidx("K" + str(i + 1))
        for i in range(self.args.P_l):
            self.action_vocab.toidx("P" + str(i + 1) + "(")
        for i in range(self.args.K_l):
            self.action_vocab.toidx("K" + str(i + 1) + "(")
        self.action_vocab.toidx("CARD_NUMBER")
        self.action_vocab.toidx("TIME_NUMBER")
        self.action_vocab.toidx(")")
        self.action_vocab.read_file(self.args.action_dict_path)
        self.action_vocab.freeze()

    def load_data(self, opt):
        if opt == 'train':
            self.inputs, self.seps = read_input(self.args.train_input)
            self.combs = [get_same_lemma(x) for x in zip(self.inputs, self.seps)]
            self.single_dict, self.word_dict, self.word_vocab = get_singleton_dict(self.inputs, self.word_vocab)
            self.instances, self.word_vocab, self.char_vocab, self.extra_vocab = input2instance(self.inputs,
                                                                                                self.word_vocab,
                                                                                                self.char_vocab,
                                                                                                self.pre_train,
                                                                                                self.extra_vocab,
                                                                                                self.word_dict,
                                                                                                self.args, "train")
            self.word_vocab.freeze()
            self.char_vocab.freeze()
            self.extra_vocab.freeze()
            self.summary()

            self.trees = read_tree(self.args.train_action)
            self.actions = tree2action(self.trees, self.action_vocab)

            self.enc_word_graph = construct_graph_maskwords(self.args.train_graph, undirected=True)
            self.dec_str_graph = construct_graph_dec_str(self.actions, undirected=True)

        elif opt == 'dev':
            self.word_vocab.read_file(self.args.model_path + "/word.list")
            self.extra_vocab.read_file(self.args.model_path + "/extra" + ".list")
            self.word_vocab.freeze()
            self.extra_vocab.freeze()
            if self.args.beam_size == 0:
                self.inputs, self.seps = read_input(self.args.dev_input)
            else:
                self.inputs, self.seps = read_input_test(self.args.dev_input)
            self.combs = [get_same_lemma(x) for x in zip(self.inputs, self.seps)]
            self.instances, self.word_vocab, self.char_vocab, self.extra_vocab = input2instance(self.inputs,
                                                                                                self.word_vocab,
                                                                                                self.char_vocab,
                                                                                                self.pre_train,
                                                                                                self.extra_vocab, {},
                                                                                                self.args, "dev")

            self.trees = read_tree(self.args.dev_action)
            self.actions = tree2action(self.trees, self.action_vocab)

            self.enc_word_graph = construct_graph_maskwords(self.args.dev_graph, undirected=True)
            self.dec_str_graph = construct_graph_dec_str(self.actions, undirected=True)

        elif opt == 'test':
            self.word_vocab.read_file(self.args.model_path + "/word.list")
            self.extra_vocab.read_file(self.args.model_path + "/extra" + ".list")
            self.word_vocab.freeze()
            self.extra_vocab.freeze()
            if self.args.beam_size == 0:
                self.inputs, self.seps = read_input(self.args.test_input)
            else:
                self.inputs, self.seps = read_input_test(self.args.test_input)
            self.combs = [get_same_lemma(x) for x in zip(self.inputs, self.seps)]
            self.instances, self.word_vocab, self.char_vocab, self.extra_vocab = input2instance(self.inputs,
                                                                                                self.word_vocab,
                                                                                                self.char_vocab,
                                                                                                self.pre_train,
                                                                                                self.extra_vocab,
                                                                                                {}, self.args,
                                                                                                "dev")
            self.trees = read_tree(self.args.test_action)
            self.actions = tree2action(self.trees, self.action_vocab)

            self.enc_word_graph = construct_graph_maskwords(self.args.test_graph, undirected=True)
            self.dec_str_graph = construct_graph_dec_str(self.actions, undirected=True)

    def summary(self):
        print("word vocabulary size:", self.word_vocab.size())
        self.word_vocab.dump(self.args.model_path + "/word.list")
        print("char vocabulary size:", self.char_vocab.size())
        if self.args.use_char:
            self.char_vocab.dump(self.args.model_path + "/char.list")
        print("pre_train vocabulary size:", self.pre_train.size())

        print("action vocabulary size:", self.action_vocab.size())
        print("extra vocabulary size:", self.extra_vocab.size())
        self.extra_vocab.dump(self.args.model_path + "/extra.list")
