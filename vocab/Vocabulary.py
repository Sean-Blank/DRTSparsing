from modules.utils import *


class Vocabulary:
	def __init__(self, UNK=True):
		if UNK:
			self._tti = {"<UNK>":0}
			self._itt = ["<UNK>"]
		else:
			self._tti = {}
			self._itt = []
		self._frozen = False

	def freeze(self):
		self._frozen = True

	def unfreeze(self):
		self._frozen = False

	def read_file(self, filename):
		with open(filename, "r", encoding='utf-8') as r:
			while True:
				l = r.readline().strip()
				if l.startswith("###"):
					continue
				if not l:
					break
				self.toidx(l)

	def toidx(self, tok):
		if tok in self._tti:
			return self._tti[tok]

		if self._frozen == False:
			self._tti[tok] = len(self._itt)
			self._itt.append(tok)
			return len(self._itt) - 1
		else:
			return 0
	def totok(self, idx):
		assert idx < self.size(), "Out of Vocabulary"
		return self._itt[idx]
	def size(self):
		return len(self._tti)
	def dump(self, filename):
		with open(filename, "w", encoding='utf-8') as w:
			for tok in self._itt:
				w.write(tok+"\n")
			w.flush()
			w.close()


def create_vocab(args):
	word_vocab = Vocabulary()
	char_vocab = Vocabulary()
	action_vocab = Vocabulary(UNK=False)
	extra_vocab = Vocabulary()

	action_vocab.toidx("<START>")
	action_vocab.toidx("<END>")
	for i in range(args.X_l):
		action_vocab.toidx("X" + str(i + 1))
	for i in range(args.E_l):
		action_vocab.toidx("E" + str(i + 1))
	for i in range(args.S_l):
		action_vocab.toidx("S" + str(i + 1))
	for i in range(args.T_l):
		action_vocab.toidx("T" + str(i + 1))
	for i in range(args.P_l):
		action_vocab.toidx("P" + str(i + 1))
	for i in range(args.K_l):
		action_vocab.toidx("K" + str(i + 1))
	for i in range(args.P_l):
		action_vocab.toidx("P" + str(i + 1) + "(")
	for i in range(args.K_l):
		action_vocab.toidx("K" + str(i + 1) + "(")
	action_vocab.toidx("CARD_NUMBER")
	action_vocab.toidx("TIME_NUMBER")
	action_vocab.toidx(")")
	action_vocab.read_file(args.action_dict_path)
	action_vocab.freeze()

	return word_vocab, char_vocab, action_vocab, extra_vocab


def get_instance(args, word_vocab, char_vocab, extra_vocab, pre_train, opt):

	if opt == 'train':
		train_input, train_sep = read_input(args.train_input)
		train_comb = [get_same_lemma(x) for x in zip(train_input, train_sep)]
		singleton_idx_dict, word_dict, word_vocab = get_singleton_dict(train_input, word_vocab)
		train_instance, word_vocab, char_vocab, extra_vocab = input2instance(train_input, word_vocab, char_vocab, pre_train, extra_vocab, word_dict, args, "train")
		word_vocab.freeze()
		char_vocab.freeze()
		extra_vocab.freeze()
		print("word vocabulary size:", word_vocab.size())
		word_vocab.dump(args.model_path_base + "/word.list")
		print("char vocabulary size:", char_vocab.size())
		if args.use_char:
			char_vocab.dump(args.model_path_base + "/char.list")
		print("pretrain vocabulary size:", pre_train.size())
		print("extra vocabulary size:", extra_vocab.size())
		extra_vocab.dump(args.model_path_base + "/extra.list")
		return train_instance, train_comb, word_vocab, char_vocab, extra_vocab
	elif opt == 'dev':
		dev_input, dev_sep = read_input(args.dev_input)
		dev_comb = [get_same_lemma(x) for x in zip(dev_input, dev_sep)]
		dev_instance, word_vocab, char_vocab, extra_vocab = input2instance(dev_input, word_vocab, char_vocab, pre_train, extra_vocab, {}, args, "dev")
		return dev_instance, dev_comb
	elif opt == 'test':
		test_input, test_sep = read_input(args.test_input)
		test_comb = [get_same_lemma(x) for x in zip(test_input, test_sep)]
		test_instance, word_vocab, char_vocab, extra_vocab = input2instance(test_input, word_vocab, char_vocab, pre_train, extra_vocab, {}, args, "dev")
		return test_instance, test_comb


def get_action(args, action_vocab, opt):
	if opt == 'train':
		train_output = read_tree(args.train_action)
		train_action = tree2action(train_output, action_vocab)
		print("action vocabulary size:", action_vocab.size())
		return train_action
	elif opt == 'dev':
		dev_output = read_tree(args.dev_action)
		dev_action = tree2action(dev_output, action_vocab)
		return dev_action
	elif opt == 'test':
		test_output = read_tree(args.test_action)
		test_action = tree2action(test_output, action_vocab)
		return test_action

