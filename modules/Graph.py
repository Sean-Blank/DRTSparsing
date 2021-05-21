import torch
import torch.nn as nn
#from torch_geometric.nn import GCNConv, GATConv
from nltk.corpus import stopwords
import torch.nn.functional as F


class GraphNet(nn.Module):
    def __init__(self, args, opt):
        super(GraphNet, self).__init__()
        self.opt = opt
        self.args = args
        self.gnn_type = self.args.gnn_type
        if self.opt == 'sent':
            self.fea_dim = self.args.bilstm_hidden_dim * self.args.bilstm_layer_num
        elif self.opt == 'word':
            self.fea_dim = self.args.bilstm_hidden_dim * self.args.bilstm_layer_num
        elif self.opt == 'dec_step1':
            self.fea_dim = self.args.bilstm_hidden_dim * self.args.bilstm_layer_num
        elif self.opt == 'dec_step2':
            self.fea_dim = self.args.bilstm_hidden_dim * self.args.bilstm_layer_num

        self.gnn_layers = nn.ModuleList()
        self.num_layers = self.args.gnn_layer
        if self.gnn_type == 'gcn':
            pass
            # for i in range(self.num_layers):
            #     #self.gnn_layers.append(GCNConv(self.fea_dim, self.fea_dim))
        elif self.gnn_type == 'gat':
            self.hidden_dim = self.fea_dim

            # hidden layers
            # for l in range(self.num_layers):
            #     # due to multi-head, the in_dim = num_hidden * num_heads
            #     self.gnn_layers.append(GATConv(self.fea_dim,
            #                                    self.fea_dim,
            #                                    self.args.heads,
            #                                    concat=False))

        self.dropout = nn.Dropout(self.args.dropout_gnn)

    def forward(self, x, edge_index, train):
        # if self.args.dropout_gnn > 0:
        #     x = F.dropout(x, p=self.args.dropout_gnn, training=self.training)
        edge_index = edge_index.transpose(0, 1)
        if self.gnn_type == 'gcn':
            for i in range(self.num_layers):
                x = F.relu(self.gnn_layers[i](x, edge_index))
        elif self.gnn_type == 'gat':
            for i in range(self.num_layers):
                x = F.elu(self.gnn_layers[i](x, edge_index))
        return x


def construct_graph_cooccurence(docs, seps):

    stop_words = stopwords.words('english')
    for w in ['!', ',', '.', '?']:
        stop_words.append(w)
    lemma = []
    for doc, sep in zip(docs, seps):
        lemma.append([])
        for i in range(len(sep)):
            if i != len(sep) - 1:

                lemma[-1].append([w for w in doc[1][(sep[i] + 1):sep[i + 1]] if w not in stop_words])
            elif i == len(sep) - 1:
                lemma[-1].append([w for w in doc[1][(sep[i]+1):-1] if w not in stop_words])
    edge = []
    weight = []
    for sents in lemma:
        edge.append([])
        weight.append([])
        for i in range(len(sents)):
            for j in range(len(sents)):
                if i != j:
                    same_wprd = len([w for w in sents[i] if w in sents[j]])
                    if same_wprd > 0:
                        edge[-1].append([i, j])
                        weight[-1].append(same_wprd)

    return edge, weight


def construct_graph_allwords(train_instance):
    doc_len = len(train_instance[0])
    edge_index = []
    for i in range(doc_len):
        for j in range(doc_len):
            edge_index.append([i, j])
    return edge_index


def construct_graph_allsents(sent_rep):
    doc_len = sent_rep.shape[0]
    edge_index = []
    for i in range(doc_len):
        for j in range(doc_len):
            if i != j:
                edge_index.append([i, j])

    return edge_index


def construct_graph_maskwords(train_graph_path, undirected):
    with open(train_graph_path, 'r', encoding='utf-8') as f:
        train_graph = [[]]
        for line in f:
            if line == '\n':
                train_graph.append([])
            else:
                e = list(map(int, line.strip('\n').split(" ")[0:2]))
                train_graph[-1].append(e)
                if undirected:
                    train_graph[-1].append([e[1], e[0]])

    while len(train_graph[-1]) == 0:
        train_graph.pop()

    return train_graph


def construct_graph_dec_str(action, undirected):
    doc_num = len(action)
    edges = []
    for i in range(doc_num):
        structure_idx = action[i][0]
        structure_len = len(structure_idx)
        stack = []
        stack_idx = []
        edge = []
        for j in range(0, structure_len):
            if j == 0:
                stack.append(structure_idx[j])
                stack_idx.append(j)
            else:
                if structure_idx[j] != 449:
                    stack.append(structure_idx[j])
                    stack_idx.append(j)
                    if undirected:
                        edge.append([stack_idx[-2], stack_idx[-1]])
                        edge.append([stack_idx[-1], stack_idx[-2]])
                    else:
                        edge.append([stack_idx[-1], stack_idx[-2]])
                elif structure_idx[j] == 449:
                    stack.pop()
                    stack_idx.pop()
        if undirected:
            edge.pop(0)
            edge.pop(0)
        else:
            edge.pop(0)
        for k, e in enumerate(edge):
            edge[k] = [e[0]-1, e[1]-1]
        if len(edge) == 0:
            edge.append([0, 0])
        edges.append(edge)

    return edges


def construct_graph_dec_rel(x, undirected):
    # edges = []
    # for i in range(len(root_node_idx)):
    #     if i == len(root_node_idx) - 1:
    #         child_num = all_hidden.size()[0] - root_node_idx[i] - 1
    #         for k in range(1, child_num + 1):
    #             if undirected:
    #                 edges.append([root_node_idx[i], root_node_idx[i] + k])
    #                 edges.append([root_node_idx[i] + k, root_node_idx[i]])
    #     else:
    #         child_num = root_node_idx[i+1] - root_node_idx[i] - 1
    #         for j in range(1, child_num + 1):
    #             if undirected:
    #                 edges.append([root_node_idx[i], root_node_idx[i] + j])
    #                 edges.append([root_node_idx[i] + j, root_node_idx[i]])

    edges = []
    for i in range(1, x.shape[0]):
        if undirected:
            edges.append([0, i])
            edges.append([i, 0])
        else:
            edges.append([0, i])

    return edges


def construct_graph_evl_str(action, undirected):

    structure_idx = action
    structure_len = len(structure_idx)
    stack = []
    stack_idx = []
    edge = []
    for j in range(0, structure_len):
        if j == 0:
            stack.append(structure_idx[j])
            stack_idx.append(j)
        else:
            if structure_idx[j] != 449:
                stack.append(structure_idx[j])
                stack_idx.append(j)
                if undirected:
                    edge.append([stack_idx[-2], stack_idx[-1]])
                    edge.append([stack_idx[-1], stack_idx[-2]])
                else:
                    edge.append([stack_idx[-1], stack_idx[-2]])
            elif structure_idx[j] == 449:
                stack.pop()
                stack_idx.pop()

    if len(edge) == 0:
        edge.append([0, 0])
    return edge


