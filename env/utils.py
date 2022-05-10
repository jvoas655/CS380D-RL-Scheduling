from torch_geometric.data import Data

import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx

# import pydot
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
import re
import os
import pickle as pkl

# Heterougenous case
durations_mean_vars = [(57, 20), (52, 3), (95, 8)]

class TaskGraph(Data):

    def __init__(self, x, edge_index, edge_weights):
        Data.__init__(self, x, edge_index.to(torch.long), edge_weights)
        self.n = len(self.x)
        self.done = {}

    def render(self, root=None):
        # graph = self.data
        graph = to_networkx(Data(self.x, self.edge_index.contiguous()))
        pos = graphviz_layout(graph, prog='dot', root=root)
        # pos = graphviz_layout(G, prog='twopi')
        # plt.figure(figsize=(8, 8))
        c_opts = [(0, 0, 0), (0.5, 0.5, 0.5)]
        color = [c_opts[1] if i in self.done else c_opts[0] for i in range(self.n)]

        nx.draw_networkx_nodes(graph, pos, node_size =  3 * self.x, node_color=color)
        nx.draw_networkx_edges(graph, pos)
        plt.show()

    def remove_edges(self, node_list):
        # mask_node = torch.logical_not(isin(self.x, node_list))
        # self.x = self.x[mask_node]
        mask_edge = isin(self.edge_index[0, :], torch.tensor(node_list)) | \
                    isin(self.edge_index[1, :], torch.tensor(node_list))
        self.edge_index = self.edge_index[:, torch.logical_not(mask_edge)]
        self.edge_attr = self.edge_attr[torch.logical_not(mask_edge), :]
        for n in node_list:
            self.done[n] = True

    def add_features_descendant(self):
        n = self.n
        x = self.x
        succ_features = torch.zeros((n, 4))
        succ_features_norm = torch.zeros((n, 4))
        edges = self.edge_index
        for i in reversed(range(n)):
            succ_i = edges[1][edges[0] == i]
            feat_i = x[i] + torch.sum(succ_features[succ_i], dim=0)
            n_pred_i = torch.FloatTensor([torch.sum(edges[1] == j) for j in succ_i])
            if len(n_pred_i) == 0:
                feat_i_norm = x[i]
            else:
                feat_i_norm = x[i] + torch.sum(succ_features_norm[succ_i] / n_pred_i.unsqueeze(1).repeat((1, 4)), dim=0)
            succ_features[i] = feat_i
            succ_features_norm[i] = feat_i_norm
        return succ_features_norm, succ_features
        # return succ_features_norm, succ_features/succ_features[0]

class Node():
    def __init__(self, type):
        self.type = type


class Cluster():
    def __init__(self, nodes, communication_cost):
        """
        :param node_types:
        :param communication_cost: [(u,v,w) with w weight]
        """
        self.nodes = nodes
        self.node_state = np.zeros(nodes)
        self.communication_cost = communication_cost


    def render(self):
        edges_list = [(u, v, {"cost": w}) for (u, v, w) in enumerate(self.communication_cost)]
        colors = ["k" if node_type == 0 else "red" for node_type in self.node_types]
        G = nx.Graph()
        G.add_nodes_from(list(range(len(self.node_types))))
        G.add_edges_from(edges_list)
        pos = graphviz_layout(G)

        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, pos=pos, node_color=colors)
        nx.draw_networkx_edges(G, pos=pos)
        nx.draw_networkx_edge_labels(G, pos=pos)
        plt.show()



def succASAP(task, n, noise):
    tasktype = task.type
    i = task.barcode[1]
    j = task.barcode[2]
    k = task.barcode[3]
    listsucc = []
    if tasktype == 0:
        if i < n:
            for j in range(i + 1, n + 1, 1):
                y = (2, i, j, 0)
                listsucc.append(Task(y, noise))
        else:
            y = (4, 0, 0, 0)
            listsucc.append(Task(y))

    if tasktype == 1:
        if j < i - 1:
            y = (1, i, j + 1, 0)
            listsucc.append(Task(y))
        else:
            y = (0, i, 0, 0)
            listsucc.append(Task(y, noise))

    if tasktype == 2:
        if i <= n - 1:
            for k in range(i + 1, j):
                y = (3, k, j, i)
                listsucc.append(Task(y, noise))
            for k in range(j + 1, n + 1):
                y = (3, j, k, i)
                listsucc.append(Task(y, noise))
            y = (1, j, i, 0)
            listsucc.append(Task(y, noise))

    if tasktype == 3:
        if k < i - 1:
            y = (3, i, j, k + 1)
            listsucc.append(Task(y, noise))
        else:
            y = (2, i, j, 0)
            listsucc.append(Task(y, noise))

    return listsucc


def CPAndWorkBelow(x, n, durations):
    x_bar = x.barcode
    C = durations[0]
    S = durations[1]
    T = durations[2]
    G = durations[3]
    ReadyTasks = []
    ReadyTasks.append(x_bar)
    Seen = []
    ToVisit = []
    ToVisit.append(x_bar)
    TotalWork = durations[x_bar[0]]
    CPl = 0
    # while len(ToVisit) > 0:
    #     for t in ToVisit:
    #         for succ in succASAP(Task(t), n):
    #             if succ not in Seen:
    #                 succ = succ.barcode
    #                 TotalWork = TotalWork + durations[succ[0]]
    #                 Seen.append(succ)
    #                 ToVisit.append(succ)
    #         ToVisit.remove(t)

    tasktype = x_bar[0]
    if tasktype == 0:
        CPl = C + (n - x_bar[1]) * (T + S + C)
    if tasktype == 1:
        CPl = (x_bar[1] - x_bar[2]) * S + C + (n - x_bar[1]) * (T + S + C)
    if tasktype == 2:
        CPl = (x_bar[2] - x_bar[1] - 1) * (T + G) + (n - x_bar[2] + 1) * (T + S + C)
    if tasktype == 3:
        CPl = (x_bar[1] - x_bar[3]) * G + (x_bar[2] - x_bar[1] - 1) * (T + G) + (n - x_bar[2] + 1) * (T + S + C)

    return (CPl, TotalWork)

def _add_task(dic_already_seen, list_to_process, task):
    if task.barcode in dic_already_seen:
        pass
    else:
        dic_already_seen[task.barcode] = len(dic_already_seen)
        list_to_process.append(task)


def _add_node(dic_already_seen, list_to_process, node):
    if node in dic_already_seen:
        pass
    else:
        dic_already_seen[node] = True
        list_to_process.append(node)


def compute_graph(n, noise=False):
    root_nodes = []
    TaskList = {}
    EdgeList = []

    root_nodes.append(Task((0, 1, 0, 0), noise))
    TaskList[(0, 1, 0, 0)] = 0

    while len(root_nodes) > 0:
        task = root_nodes.pop()
        list_succ = succASAP(task, n, noise)
        for t_succ in list_succ:
            _add_task(TaskList, root_nodes, t_succ)
            EdgeList.append((TaskList[task.barcode], TaskList[t_succ.barcode]))

    # embeddings
    embeddings = [k for k in TaskList]

    data = Data(x=torch.tensor(embeddings, dtype=torch.float),
                edge_index=torch.tensor(EdgeList).t().contiguous())

    task_array = []
    for (k, v) in TaskList.items():
        task_array.append(Task(k, noise=noise))
    return TaskGraph(x=torch.tensor(embeddings, dtype=torch.float),
                edge_index=torch.tensor(EdgeList).t().contiguous(), task_list=task_array)
    # return data, task_array


def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)

def remove_nodes(edge_index, mask, num_nodes):
    r"""Removes the isolated nodes from the graph given by :attr:`edge_index`
    with optional edge attributes :attr:`edge_attr`.
    In addition, returns a mask of shape :obj:`[num_nodes]` to manually filter
    out isolated node features later on.
    Self-loops are preserved for non-isolated nodes.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (LongTensor, Tensor, BoolTensor)
    """
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=mask.device)
    assoc[mask] = torch.arange(mask.sum(), device=assoc.device)
    edge_index = assoc[edge_index]
    return edge_index

def compute_sub_graph(data, root_nodes, window):
    """
    :param data: the whole graph
    :param root_nodes: list of node numbers
    :param window: the max distance to go down from the root nodes
    :return: the sub graph with nodes at distance less than h from root_nodes
    """
    already_seen = torch.zeros(data.num_nodes, dtype=torch.bool)
    already_seen[root_nodes] = 1
    edge_list = torch.tensor([[], []], dtype=torch.long)

    i = 0
    while len(root_nodes) > 0 and i < window:
        mask = isin(data.edge_index[0], root_nodes)
        list_succ = data.edge_index[1][mask]
        list_pred = data.edge_index[0][mask]

        edge_list = torch.cat((edge_list, torch.stack((list_pred, list_succ))), dim=1)

        list_succ = torch.unique(list_succ)

        list_succ = list_succ[already_seen[list_succ] == 0]
        already_seen[list_succ] = 1
        root_nodes = list_succ
        i += 1

    assoc = torch.full((len(data.x),), -1, dtype=torch.long)
    assoc[already_seen] = torch.arange(already_seen.sum())

    node_num = torch.nonzero(already_seen)
    new_x = data.x[already_seen]
    new_edge_index = remove_nodes(data.edge_index, already_seen, len(data.x))
    mask_edge = (new_edge_index != -1).all(dim=0)
    new_edge_index = new_edge_index[:, mask_edge]
    new_edge_attrs = data.edge_attr[mask_edge, :]

    return TaskGraph(new_x, new_edge_index, new_edge_attrs), node_num

def ggen_roue(n_vertex, n_edges, as_density = False):
    if (n_vertex % 2 == 0):
        max_edges = (n_vertex - 1) * (n_vertex // 2)
    else:
        max_edges = n_vertex * (n_vertex // 2)
    if (as_density):
        n_edges = int(n_edges * max_edges)
    n_edges = int(min(max_edges, n_edges))
    node_duration_specs = np.take(durations_mean_vars, np.random.choice(len(durations_mean_vars), size = (n_vertex, 1)), axis = 0).squeeze()
    node_attrs = np.random.normal(node_duration_specs[:, 0], node_duration_specs[:, 1]).reshape(-1, 1)
    node_edges = np.zeros((2, n_edges))
    edge_attrs = np.clip(np.random.normal(10, 5, (n_edges, 1)), a_min=0.1, a_max=None)
    sort_nodes = np.arange(n_vertex)
    np.random.shuffle(sort_nodes)
    prior = set()
    edge_count = 0
    while (edge_count < n_edges):
        e1 = np.random.randint(0, n_vertex - 1)
        e2 = np.random.randint(e1+1, n_vertex)
        if ((e1, e2) in prior):
            continue
        prior |= {(e1, e2)}
        node_edges[:, edge_count] = [sort_nodes[e1], sort_nodes[e2]]
        edge_count += 1
    return TaskGraph(x=torch.tensor(node_attrs, dtype=torch.float),
                edge_index=torch.tensor(node_edges), edge_weights=torch.tensor(edge_attrs))

if __name__ == "__main__":
    import time
    t = time.time()
    g = ggen_roue(100, 0.07, True)
    print(time.time() - t)
    g.render()
