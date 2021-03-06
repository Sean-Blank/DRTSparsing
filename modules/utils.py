# -*- coding: utf-8 -*-
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import torch
import torch.optim as optim

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def get_lemmas(tokens):
    lemmatizer = WordNetLemmatizer()
    pos_tokens = [nltk.pos_tag(tokens)]
    lemmas = []
    for pos in pos_tokens[0]:
        word, pos_tag = pos
        pos_tag = get_wordnet_pos(pos_tag)
        if pos_tag == None:
            lemmas.append(word.lower())
        else:
            lemmas.append(lemmatizer.lemmatize(word.lower(), pos_tag))
    # lemmas[-1] = lemmas[-1].encode("utf-8")
    return lemmas


def read_input_test(filename):
    data = [[]]
    seps = [[]]

    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":
                data.append([])
                seps.append([])
            else:
                if line[0] == "#":
                    continue
                data[-1].append(["<s>"] + line.split() + ["</s>"])
                if len(seps[-1]) == 0:
                    for i, w in enumerate(data[-1][-1]):
                        if w in ["<s>", "</s>", "|||"]:
                            seps[-1].append(i)
    if len(data[-1]) == 0:
        data.pop()
        seps.pop()

    # print data[0]
    # resign lem
    for i, t in enumerate(data):
        w, l = t
        words = " ".join(w[1:-1]).split("|||")
        lem = ""
        for word in words:
            lem += " ".join(get_lemmas(word.split())) + " ||| "
        lem = ["<s>"] + lem[:-5].split() + ["</s>"]
        data[i][1] = lem
    return data, seps


def read_input(filename):
    data = [[]]
    seps = [[]]
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":
                data.append([])
                seps.append([])
            else:
                if line[0] == "#":
                    continue
                data[-1].append(["<s>"] + line.split() + ["</s>"])
                if len(seps[-1]) == 0:
                    for i, w in enumerate(data[-1][-1]):
                        if w in ["<s>", "</s>", "|||"]:
                            seps[-1].append(i)
        if len(data[-1]) == 0:
            data.pop()
            seps.pop()

    return data, seps


def get_singleton_dict(train_input, word_v):
    d = {}
    singleton_idx_dict = {}
    word_dict = {}
    for instance in train_input:
        for w in instance[0]:
            if w in d:
                d[w] += 1
            else:
                d[w] = 1
            word_v.toidx(w)

    # for key in d.keys():
    for key in list(d.keys()):
        if d[key] == 1:
            singleton_idx_dict[word_v.toidx(key)] = 1
        else:
            word_dict[key] = 1
    return singleton_idx_dict, word_dict, word_v


def unkify(token, word_dict, lang):
    if len(token.rstrip()) == 0:
        return '<UNK>'
    elif not (token.rstrip() in word_dict):
        if lang == "ch":
            return '<UNK>'
        numCaps = 0
        hasDigit = False
        hasDash = False
        hasLower = False
        for char in token.rstrip():
            if char.isdigit():
                hasDigit = True
            elif char == '-':
                hasDash = True
            elif char.isalpha():
                if char.islower():
                    hasLower = True
                elif char.isupper():
                    numCaps += 1
        result = '<UNK>'
        lower = token.rstrip().lower()
        ch0 = token.rstrip()[0]
        if ch0.isupper():
            if numCaps == 1:
                result = result + '-INITC'
                if lower in word_dict:
                    result = result + '-KNOWNLC'
            else:
                result = result + '-CAPS'
        elif not (ch0.isalpha()) and numCaps > 0:
            result = result + '-CAPS'
        elif hasLower:
            result = result + '-LC'
        if hasDigit:
            result = result + '-NUM'
        if hasDash:
            result = result + '-DASH'
        if lower[-1] == 's' and len(lower) >= 3:
            ch2 = lower[-2]
            if not (ch2 == 's') and not (ch2 == 'i') and not (ch2 == 'u'):
                result = result + '-s'
        elif len(lower) >= 5 and not (hasDash) and not (hasDigit and numCaps > 0):
            if lower[-2:] == 'ed':
                result = result + '-ed'
            elif lower[-3:] == 'ing':
                result = result + '-ing'
            elif lower[-3:] == 'ion':
                result = result + '-ion'
            elif lower[-2:] == 'er':
                result = result + '-er'
            elif lower[-3:] == 'est':
                result = result + '-est'
            elif lower[-2:] == 'ly':
                result = result + '-ly'
            elif lower[-3:] == 'ity':
                result = result + '-ity'
            elif lower[-1] == 'y':
                result = result + '-y'
            elif lower[-2:] == 'al':
                result = result + '-al'
        return result
    else:
        return token.rstrip()


def input2instance(train_input, word_v, char_v, pretrain, extra_vl, word_dict, args, op):
    train_instance = []

    for instance in train_input:
        # lexicon representation
        train_instance.append([])
        train_instance[-1].append([])  # word
        train_instance[-1].append([])  # char
        train_instance[-1].append([])  # pretrain
        train_instance[-1].append([])  # typed UNK
        for w in instance[0]:
            idx = word_v.toidx(w)
            if op == "train":
                train_instance[-1][0].append(idx)
                idx = word_v.toidx(unkify(w, word_dict, "en"))
                train_instance[-1][3].append(idx)
            elif op == "dev":
                if idx == 0:  # UNK
                    idx = word_v.toidx(unkify(w, word_dict, "en"))
                train_instance[-1][0].append(idx)

            if args.use_char:
                train_instance[-1][1].append([])
                for c in list(w):
                    idx = char_v.toidx(c)
                    train_instance[-1][1][-1].append(idx)
            if args.pretrain_path:
                idx = pretrain.toidx(w.lower())
                train_instance[-1][2].append(idx)
        # extra representatoin
        for i, extra_info in enumerate(instance[1:], 0):

            train_instance[-1].append([])
            for t in extra_info:
                idx = extra_vl.toidx(t)
                train_instance[-1][-1].append(idx)

    return train_instance, word_v, char_v, extra_vl


def read_tree(filename):
    data = []
    for line in open(filename):
        line = line.strip()
        if line[0] == "#":
            continue
        data.append(bracket2list(line.split()))
    return data


def bracket2list(bracket):
    stack = []
    for tok in bracket:
        if tok[-1] == "(":
            stack.append([tok])
        elif tok == ")":
            if len(stack) != 1:
                back = stack.pop()
                stack[-1].append(back)
        else:
            stack[-1].append(tok)
    assert len(stack) == 1
    return stack[0]


def tree2action(output, actn_v):
    actions = []
    for line in output:
        # print line
        actions.append(get_struct_rel_var(line, actn_v))
    return actions


def is_struct(tok):
    if tok in ["DRS(", "SDRS(", "NOT(", "POS(", "NEC(", "IMP(", "OR(", "DUP("]:
        return True
    if re.match("^DRS-[0-9]+\($", tok):
        return True
    if re.match("^[PK][0-9]+\($", tok):
        return True
    return False


def get_struct_rel_var(tree, actn_v):
    """
    input list of string
    return list of index
    """
    current_pointer = [0]
    struct = [actn_v.toidx("<START>")]
    struct_pointer = [-1]
    relation = []
    relation_pointer = []
    variable = []
    variable_pointer = []

    def travel(root):
        parent = root[0]
        child = root[1:]
        if parent[-1] == "(":
            if is_struct(parent):
                if re.match("^DRS-[0-9]+\($", parent):
                    struct.append(actn_v.toidx("DRS("))
                    current_pointer[0] = int(parent[4:-1])
                else:
                    struct.append(actn_v.toidx(parent))
                struct_pointer.append(current_pointer[0])
                if (parent in ["DRS(", "SDRS("]) or re.match("^DRS-[0-9]+\($", parent):
                    relation.append([])
                    relation_pointer.append([])
                    for c in child:
                        if not is_struct(c[0]):
                            if re.match("^\$[0-9]+\($", c[0]):
                                relation[-1].append(c[0])
                            else:
                                relation[-1].append(actn_v.toidx(c[0]))
                            relation_pointer[-1].append(current_pointer[0])
                            variable.append([actn_v.toidx(cc) for cc in c[1:]] + [actn_v.toidx(")")])
                            variable_pointer.append([current_pointer[0] for n in variable[-1]])
                    relation[-1].append(actn_v.toidx(")"))
                    relation_pointer[-1].append(current_pointer[0])
                for c in child:
                    travel(c)
                struct.append(actn_v.toidx(")"))
                struct_pointer.append(current_pointer[0])
            else:
                pass
        # if re.match("^\$[0-9]+\($",parent):
        #	relation[-1].append(parent)
        # else:
        #	relation[-1].append(actn_v.toidx(parent))

        else:
            print(parent)
            assert False

    travel(tree)
    # struct.append(actn_v.toidx("<END>"))
    assert len(struct) == len(struct_pointer)  # <START>
    assert len(variable) == len(variable_pointer)
    for i in range(len(variable)):
        assert len(variable[i]) == len(variable_pointer[i])
    assert len(relation) == len(relation_pointer)
    for i in range(len(relation)):
        assert len(relation[i]) == len(relation_pointer[i])
    # revise pointer for six scopes and p k scopes
    for i, idx in enumerate(struct):
        if actn_v.totok(idx) == "DRS(":
            j = i - 1
            while j >= 0 and actn_v.totok(struct[j]) != ")" and actn_v.totok(struct[j]) != "DRS(":
                struct_pointer[j] = struct_pointer[i]
                j -= 1
    return [struct, relation, variable, struct_pointer, relation_pointer, variable_pointer]


def output2action(train_output, actn_v):
    """
    get action index in actn_v
    """
    train_action = []
    for output in train_output:
        train_action.append([])
        for a in output:
            idx = actn_v.toidx(a)
            train_action[-1].append(idx)
    return train_action, actn_v


def output2action_cpy(train_output, actn_v):
    """
    get action index in actn_v if action is not for copy
    remain action string  if action is for copy,
    """
    import re
    cpy1_p = re.compile("^\$[0-9]+$")
    train_action = []
    for output in train_output:
        train_action.append([])
        for a in output:
            if a in ["B", "X", "E", "S", "T", "P"]:
                train_action[-1].append([a, actn_v.toidx(a)])
            elif cpy1_p.match(a):
                train_action[-1].append(a)
            else:
                idx = actn_v.toidx(a)
                train_action[-1].append(idx)
    return train_action


def get_same_lemma(packed):
    """
    get matrix that indicates which lemmas are the same
    """
    lemmas = packed[0][1]
    seps = packed[1]
    combs = []
    for j in range(len(seps) - 1):
        s = seps[j]
        e = seps[j + 1]
        comb = []
        past_lemmas = []
        for i, li in enumerate(lemmas[s + 1:e]):
            if li in past_lemmas:
                continue
            past_lemmas.append(li)
            comb.append([])
            for k, lk in enumerate(lemmas[s + 1:e]):
                if lk == li:
                    comb[-1].append(k)
        # 每句话中相同的单词放一起了
        combs.append(comb)
    return combs


def get_k_scope(output, actn_v):
    stack = []
    k_scope = {}
    sdrs_idx = 0
    for act in output:
        act_s = actn_v.totok(act)
        if act_s[-1] == "(":
            if act_s == "SDRS(":
                stack.append([sdrs_idx, []])
                sdrs_idx += 1
            elif re.match("^K[0-9]+\($", act_s):
                stack.append([1000 + int(act_s[1:-1]) - 1, []])
            else:
                stack.append([-1, []])
        elif actn_v.totok(act) == ")":
            b = stack.pop()
            if b[0] != -1 and b[0] < 1000:
                k_scope[b[0]] = b[1]
            if len(stack) > 0:
                stack[-1][1] = stack[-1][1] + b[1]
            if b[0] >= 1000:
                stack[-1][1].append(b[0] % 1000)
    return k_scope


def get_p_max(output, actn_v):
    p_max = -1
    for act in output:
        if re.match("^P[0-9]+\($", actn_v.totok(act)):
            p_max = max(p_max, int(actn_v.totok(act)[1:-1]) - 1)

    return p_max


def out_tree(lemmas, trees):
    lems = " ".join(lemmas[1:-1]).split("|||")
    for i, x in enumerate(lems):
        lems[i] = []
        for y in x.split():
            if y not in lems[i]:
                lems[i].append(y)
    cur = 0
    j = 0
    while j < len(trees):
        if re.match("^DRS-[0-9]+\($", trees[j]):
            cur = int(trees[j][4:-1])
            assert cur < len(lems)
        elif re.match("^\$[0-9]+\(", trees[j]):
            idx = int(trees[j][1:-1])
            trees[j] = lems[cur][idx] + "("
        j += 1
    return " ".join(trees)


def remove_node(all_hidden, root_node_idx):
    # all_hidden: 374 x 600 --> 374-26 x 1 x 600
    list_temp = list(all_hidden.split(1, 0))
    for i in range(len(root_node_idx) - 1, -1, -1):
        list_temp.pop(root_node_idx[i])
    return torch.cat(list_temp, 0).unsqueeze(1)


def optimizer(args, parameters):
    if args.optimizer.lower() == "adam":
        return optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=args.learning_rate,
                          weight_decay=args.weight_decay,)
    elif args.optimizer.lower() == "sgd":
        return optim.SGD(filter(lambda p: p.requires_grad, parameters), lr=args.learning_rate,
                         weight_decay=args.weight_decay)
    else:
        assert False, "no application for the optimizer"
