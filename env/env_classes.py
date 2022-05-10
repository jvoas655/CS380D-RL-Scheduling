import gym
from gym.spaces import Box, Dict
import string

from env.utils import *
from env.utils import compute_graph
import heft
from copy import deepcopy
import ipdb
from torch.nn import functional as F

class RDAGEnv(gym.Env):
    def __init__(self, args):

        self.observation_space = Dict
        self.action_space = "Graph"
        self.args = args
        self.time = 0
        self.num_steps = 0

        if self.args.env_type == 'RouE':
            self.task_data = ggen_roue(self.args.task_nodes, self.args.edges, self.args.as_density)
        elif self.args.env_type == 'RouP':
            self.task_data = None
            raise NotImplementedError(self.args.env_type)
        elif self.args.env_type == 'LevelP':
            self.task_data = None
            raise NotImplementedError(self.args.env_type)
        elif self.args.env_type == 'FinFout':
            self.task_data = None
            raise NotImplementedError(self.args.env_type)
        elif self.args.env_type == 'RandOrd':
            self.task_data = None
            raise NotImplementedError(self.args.env_type)
        elif self.args.env_type == 'MarkovChain':
            self.task_data = None
            raise NotImplementedError(self.args.env_type)
        elif self.args.env_type == 'MarkovChainEven':
            self.task_data = None
            raise NotImplementedError
        else:
            raise EnvironmentError('not implemented')
        self.num_nodes = self.task_data.num_nodes
        self.sum_task = torch.sum(self.task_data.x, dim=0)
        self.norm_desc_features = self.task_data.add_features_descendant()[0] / self.sum_task
        self.cluster = Cluster(self.args.processor_nodes, communication_cost=np.logical_not(np.identity(self.args.processor_nodes)).astype(int))
        self.running = -1 * np.ones(self.args.processor_nodes)  # array of task number
        self.running_task2proc = {}
        self.ready_proc = np.zeros(self.args.processor_nodes)  # for each processor, the time where it becomes available
        self.ready_tasks = []
        self.task_comun = {}
        self.processor_perf = np.random.random(self.args.processor_nodes) # Start node performance as random between [0,1]
        self.processed = {}
        self.compeur_task = 0
        self.last_perf_update_time = 0
        self.current_proc = 0
        self.comm_sum = 0
        self.comp_sum = 0
        self.comm_factors = []
        self.utilization = np.zeros(self.args.processor_nodes)
        self.wait_free_utilization = np.zeros(self.args.processor_nodes)
        self.task_to_comm_time = {}
        self.critic_path_duration = None
        self.total_work_normalized = None
        self.history = np.array([-1] * self.num_nodes)
        new_ready_tasks = torch.arange(0, self.num_nodes)[torch.logical_not(isin(torch.arange(0, self.num_nodes), self.task_data.edge_index[1, :]))]
        self.ready_tasks = new_ready_tasks.tolist()


        # compute heft
        string_cluster = string.printable[:self.args.processor_nodes]
        dic_heft = {}
        for edge in np.array(self.task_data.edge_index.t()):
            dic_heft[edge[0]] = dic_heft.get(edge[0], ()) + (edge[1],)

        def compcost(job, agent):
            idx = string_cluster.find(agent)
            duration = self.task_data.x[idx]
            return duration

        def commcost(ni, nj, A, B):
            edge = np.logical_and(self.task_data.edge_index[0] == ni, self.task_data.edge_index[1] == nj)
            if (np.any(np.where(edge > 0, True, False))):
                return self.cluster.communication_cost[int(A)][int(B)] * self.task_data.edge_attr[np.argmax(edge).item()].item()
            else:
                return float("inf")
        orders, jobson = heft.schedule(dic_heft, string_cluster, compcost, commcost)

        self.heft_time = max([v[-1].end.item() for v in orders.values() if len(v) > 0])
    def reset(self):
        # self.task_data = random_ggen_fifo(self.n, self.max_in, self.max_out, self.noise)
        if self.args.env_type == 'RouE':
            self.task_data = ggen_roue(self.args.task_nodes, self.args.edges, self.args.as_density)
        elif self.args.env_type == 'RouP':
            self.task_data = None
            raise NotImplementedError(self.args.env_type)
        elif self.args.env_type == 'LevelP':
            self.task_data = None
            raise NotImplementedError(self.args.env_type)
        elif self.args.env_type == 'FinFout':
            self.task_data = None
            raise NotImplementedError(self.args.env_type)
        elif self.args.env_type == 'RandOrd':
            self.task_data = None
            raise NotImplementedError(self.args.env_type)
        elif self.args.env_type == 'MarkovChain':
            self.task_data = None
            raise NotImplementedError(self.args.env_type)
        elif self.args.env_type == 'MarkovChainEven':
            self.task_data = None
            raise NotImplementedError
        else:
            raise EnvironmentError('not implemented')
        self.time = 0
        self.num_steps = 0
        self.running = -1 * np.ones(self.args.processor_nodes).astype(int)
        self.running_task2proc = {}
        self.ready_proc = np.zeros(self.args.processor_nodes)
        self.task_comun = {}
        self.processor_perf = np.clip(np.random.random(self.args.processor_nodes), a_min=0.1, a_max=1) # Start node performance as random between [0,1]
        self.last_perf_update_time = 0
        self.ready_tasks = []
        self.current_proc = 0
        self.history = np.array([-1] * self.num_nodes)
        self.comm_sum = 0
        self.comp_sum = 0
        self.comm_factors = []
        self.utilization = np.zeros(self.args.processor_nodes)
        self.wait_free_utilization = np.zeros(self.args.processor_nodes)
        self.task_to_comm_time = {}
        # compute initial doable tasks

        new_ready_tasks = torch.arange(0, self.num_nodes)[torch.logical_not(isin(torch.arange(0, self.num_nodes), self.task_data.edge_index[1, :]))]
        self.ready_tasks = new_ready_tasks.tolist()
        self.compeur_task = 0

        # compute heft
        string_cluster = string.printable[:self.args.processor_nodes]
        dic_heft = {}
        for edge in np.array(self.task_data.edge_index.t()):
            dic_heft[edge[0]] = dic_heft.get(edge[0], ()) + (edge[1],)

        def compcost(job, agent):
            idx = string_cluster.find(agent)
            duration = self.task_data.x[idx]
            return duration

        def commcost(ni, nj, A, B):
            edge = np.logical_and(self.task_data.edge_index[0] == ni, self.task_data.edge_index[1] == nj)
            if (np.any(np.where(edge > 0, True, False))):
                return self.cluster.communication_cost[int(A)][int(B)] * self.task_data.edge_attr[np.argmax(edge).item()].item()
            else:
                return float("inf")
        orders, jobson = heft.schedule(dic_heft, string_cluster, compcost, commcost)
        self.heft_time = max([v[-1].end.item() for v in orders.values() if len(v) > 0])

        return self._compute_state()

    def step(self, action, render_before=False, render_after=False, speed=False):
        """
        first implementation, with only [-1, 0, ..., T] actions
        :param action: -1: does nothing. t: schedules t on the current available processor
        :return: next_state, reward, done, info
        """

        self.num_steps += 1

        if action != -1:
            self.compeur_task += 1
            self.history[self.ready_tasks[action]] = self.current_proc
        sched_reward = self._choose_task_processor(action, self.current_proc)

        if render_before:
            self.render()

        done, step_reward = self._go_to_next_action()

        if render_after and not speed:
            self.render()

        reward = -self.time if done else -sched_reward
        if (done):
            print("HEFT", self.heft_time, "TIME", self.time, "COMP", self.comp_sum, "COMM", self.comm_sum)
            print("UTILS", self.utilization / self.time)
            print("WF UTILS", self.wait_free_utilization / self.time)
            print("COMM DISCOUNT", np.mean(self.comm_factors))


        info = {'episode': {'r': reward, 'length': self.num_steps, 'time': self.time}, 'bad_transition': False}

        if speed:
            return 0, reward, done, info

        return self._compute_state(), reward, done, info

    def _find_available_proc(self):
        self.current_proc += 1
        while (self.current_proc < self.args.processor_nodes) and (self.running[self.current_proc] > -1):
            self.current_proc += 1
        if self.current_proc == self.args.processor_nodes:
            # no new proc available
            self.current_proc == -1
            return False
        return True

    def _forward_in_time(self):

        if len(self.ready_proc[self.ready_proc > self.time]) > 0:
            min_time = np.min(self.ready_proc[self.ready_proc > self.time])
        else:
            min_time = 0

        self.time = min_time
        while (self.time - self.last_perf_update_time > 1):
            self.last_perf_update_time += 1
            self.processor_perf = np.clip(self.processor_perf + np.random.normal(0, 0.01, self.processor_perf.shape), a_min=0.1, a_max=1)
        self.ready_proc[self.ready_proc < self.time] = self.time

        tasks_finished = self.running[np.logical_and(self.ready_proc == self.time, self.running > -1)].copy()
        self.running[self.ready_proc == self.time] = -1
        # compute successors of finished tasks

        mask = isin(self.task_data.edge_index[0], torch.tensor(tasks_finished))
        list_succ = self.task_data.edge_index[1][mask]
        list_succ = torch.unique(list_succ)
        mask = isin(list_succ, self.task_data.edge_index[1][torch.logical_not(mask)])
        list_succ = list_succ[torch.logical_not(mask)]
        for task in tasks_finished:
            task_mask = isin(self.task_data.edge_index[0], task)
            requires_succ = self.task_data.edge_index[1][task_mask]
            for succ_task_ind in range(requires_succ.shape[0]):
                succ_task = requires_succ[succ_task_ind].tolist()
                edge_mask = isin(self.task_data.edge_index[0, :], task) & \
                            isin(self.task_data.edge_index[1, :], succ_task)
                if (succ_task in self.task_comun):
                    self.task_comun[succ_task].append((self.running_task2proc[task], self.task_data.edge_attr[edge_mask, 0]))
                else:
                    self.task_comun[succ_task] = [(self.running_task2proc[task], self.task_data.edge_attr[edge_mask, 0])]
            del self.running_task2proc[task]
            if (task in self.task_comun):
                del self.task_comun[task]
        # remove nodes
        self.task_data.remove_edges(tasks_finished)
        #self.task_data.render()

        # compute new available tasks
        #new_ready_tasks = list_succ[torch.logical_not(isin(list_succ, self.task_data.edge_index[1, :]))]
        self.ready_tasks += list_succ.tolist()
        self.current_proc = 0
        comm_reward = 0
        for task in tasks_finished:
            comm_reward -= self.task_to_comm_time[task]
        return comm_reward

    def _go_to_next_action(self):
        rewards = []
        while len(self.ready_tasks) == 0:
            rewards.append(self._forward_in_time())
            if self._isdone():
                return True, np.sum(rewards)
        while (True):
            if (self._find_available_proc()):
                return False, np.sum(rewards)
            else:
                rewards.append(self._forward_in_time())

    def _choose_task_processor(self, action, processor):
        # assert action in self.ready_tasks

        if action != -1:
            #ipdb.set_trace()
            if (self.ready_tasks[action] in self.task_comun):
                comun_costs = max([self.cluster.communication_cost[i[0], processor] * i[1] for i in self.task_comun[self.ready_tasks[action]]])
            else:
                comun_costs = 0
            if (comun_costs > 0):
                self.comm_factors.append(1)
            else:
                self.comm_factors.append(0)
            self.task_to_comm_time[self.ready_tasks[action]] = comun_costs
            self.comm_sum += comun_costs
            self.comp_sum += self.task_data.x[self.ready_tasks[action]] / self.processor_perf[processor]

            self.ready_proc[processor] += comun_costs + self.task_data.x[self.ready_tasks[action]] / self.processor_perf[processor]
            self.utilization[processor] += comun_costs + self.task_data.x[self.ready_tasks[action]] / self.processor_perf[processor]
            self.wait_free_utilization[processor] += self.task_data.x[self.ready_tasks[action]] / self.processor_perf[processor]
            self.running_task2proc[self.ready_tasks[action]] = processor
            self.running[processor] = self.ready_tasks[action]
            self.ready_tasks.remove(self.ready_tasks[action])
            return comun_costs

    def _calc_dependent_task(self, size):
        #TODO: find previous task given the next task that is going to execute
        '''

        :return:
        '''
        store_dict = {}
        for t in self.ready_tasks:
            store_dict[t] = None

        node_task_mx = torch.tensor(np.zeros((size, self.cluster.nodes)),dtype=torch.float)
        # this might be slow
        for i in range(self.task_data.edge_index.shape[1]): # 2*len
            tail = self.task_data.edge_index[1, i]
            head = self.task_data.edge_index[0, i]
            weight = self.task_data.edge_attr[i]
            if tail in self.ready_tasks:
                node_task_mx[tail, self.history[head]] += weight
        return node_task_mx # ready_tasks * num_of_processors, meaning how many previous results are stored on each processor

    def _compute_state(self):
        visible_graph, node_num = compute_sub_graph(self.task_data,
                                          torch.tensor(np.concatenate((self.running[self.running > -1],
                                                                       self.ready_tasks)), dtype=torch.long),
                                          self.args.window)
        #history_mx = self._calc_dependent_task(node_num.shape[0])
        visible_graph.x, ready = self._compute_embeddings(node_num, visible_graph.x)
        cluster_emb = self._compute_cluster_embeddings() # size = processor_num * 3
        return {'graph': visible_graph, 'node_num': node_num, 'ready': ready, 'cluster': cluster_emb}

    def _compute_embeddings(self, tasks, time):
        ready = isin(tasks, torch.tensor(self.ready_tasks)).float()
        running = isin(tasks, torch.tensor(self.running[self.running > -1])).squeeze(-1)

        remaining_time = torch.zeros(tasks.shape[0])
        remaining_time[running] = self._remaining_time(tasks[running].squeeze(-1)).to(torch.float)
        remaining_time = remaining_time.unsqueeze(-1)

        n_succ = torch.sum((tasks == self.task_data.edge_index[0]).float(), dim=1).unsqueeze(-1) # task after this
        n_pred = torch.sum((tasks == self.task_data.edge_index[1]).float(), dim=1).unsqueeze(-1) # task before this

        # add other embeddings

        descendant_features_norm = self.norm_desc_features[tasks].squeeze(1)

        running_time = torch.tensor(time, dtype=torch.float) # Causing error
        running_time_norm = torch.norm(running_time)
        if (running_time_norm.is_nonzero()):
            running_time = torch.div(running_time, running_time_norm)
        comun_time = torch.zeros(tasks.shape[0])
        for task in range(tasks.shape[0]):
            if (ready[task].is_nonzero()):
                if (tasks[task].item() in self.task_comun):
                    comun_time[task] = max([self.cluster.communication_cost[i[0], self.current_proc] * i[1] for i in self.task_comun[tasks[task].item()]])
                else:
                    comun_time[task] = 0
        #ipdb.set_trace()
        return (torch.cat((running_time, n_succ, comun_time.unsqueeze(-1), ready, running.unsqueeze(-1).float(), remaining_time,
                           descendant_features_norm), dim=1), # history task_num * 10
                ready)
            # full size would be 16 - 3 - 4 + 10

    def _compute_cluster_embeddings(self):
        performance = torch.tensor(self.processor_perf, dtype=torch.float).unsqueeze(1)
        aval_time = torch.tensor(self.ready_proc, dtype=torch.float).unsqueeze(1)  # Causing error
        aval_time_norm = torch.norm(aval_time)
        if (aval_time_norm.is_nonzero()):
            aval_time = torch.div(aval_time, aval_time_norm)
        aval = torch.tensor(self.ready_proc == 0, dtype=torch.float).unsqueeze(1)
        cur_proc = torch.zeros(self.processor_perf.shape[0], dtype=torch.float)
        cur_proc[self.current_proc] = 1
        cur_proc = cur_proc.unsqueeze(1)
        return torch.cat((performance, aval, aval_time, cur_proc), dim=1) # cluster embedding, processor_num * 3

    def _remaining_time(self, running_tasks):
        return torch.tensor([self.ready_proc[self.running_task2proc[task.item()]] for task in running_tasks]) - self.time

    def _isdone(self):
        # return (self.task_data.edge_index.shape[-1] == 0) and (len(self.running_task2proc) == 0)
        return (self.compeur_task == self.num_nodes and (len(self.running_task2proc) == 0))

    # def _compute_embeddings(self, tasks):
    #     return NotImplementedError

    def render(self):
        raise NotImplementedError
        def color_task(task):
            colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            if task in self.running:
                time_proportion =1 - (self.ready_proc[self.running_task2proc[task]] - self.time)/\
                                  self.task_data.task_list[task].duration_cpu
                color_time = [1., time_proportion, time_proportion]
                return color_time
            elif task in self.ready_tasks:
                return colors[1]
            return colors[2]

        def color_processor(processor):
            if self.running[processor] == -1:
                return [0, 1, 0] if self.current_proc == processor else [0.7, 0.7, 0.7]
            else:
                time_proportion = (self.ready_proc[processor] - self.time) / \
                                  self.task_data.task_list[self.running[processor]].duration_cpu
            return [time_proportion, 0, 0]

        visible_graph, node_num = compute_sub_graph(self.task_data,
                                          torch.tensor(np.concatenate((self.running[self.running > -1],
                                                                       self.ready_tasks)), dtype=torch.long),
                                          self.window)
        plt.figure(figsize=(8 , 8))
        plt.suptitle('time: {}'.format(self.time))
        plt.subplot(121)
        plt.box(on=None)
        visible_graph.render(root=list(self.running[self.running > -1]))
        # plt.title('time: {}'.format(self.time))
        # plt.show()

        plt.subplot(122)
        plt.box(on=None)
        graph = to_networkx(Data(visible_graph.x, visible_graph.edge_index.contiguous()))
        pos = graphviz_layout(graph, prog='dot', root=None)
        # pos = graphviz_layout(G, prog='tree')
        node_color = [color_task(task[0].item()) for task in node_num]
        # plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(graph, pos, node_color=node_color)
        nx.draw_networkx_edges(graph, pos)
        labels = {}
        for i, task in enumerate(node_num):
            if task[0].item() in self.ready_tasks:
                labels[i] = task[0].item()
        nx.draw_networkx_labels(graph, pos, labels, font_size=16)
        # plt.title('time: {}'.format(self.time))
        plt.show()

        # Cluster
        edges_list = [(u, v, {"cost": self.cluster.communication_cost[u, v]}) for u in range(self.p) for v in range(self.p) if u != v]
        colors = [color_processor(p) for p in range(self.p)]
        G = nx.Graph()
        G.add_nodes_from(list(range(len(self.cluster.node_types))))
        G.add_edges_from(edges_list)
        pos = graphviz_layout(G)
        node_labels = {}
        for i, node_type in enumerate(self.cluster.node_types):
            node_labels[i] = ["CPU", "GPU"][node_type]

        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, pos=pos, node_color=colors, node_size=1000)
        nx.draw_networkx_edges(G, pos=pos)
        nx.draw_networkx_edge_labels(G, pos=pos)
        nx.draw_networkx_labels(G, pos, node_labels, font_size=16)
        plt.show()

    def visualize_schedule(self, figsize=(80, 30), fig_file=None, flip=False):
        raise NotImplementedError
        def get_data(env):
            P = env.p
            Processed = env.processed
            for k, v in Processed.items():
                Processed[k] = [int(v[0]), int(v[1])]

            # makespan should be dicrete and durations should be discretized
            makespan = int(env.time)
            data = np.ones((P, makespan)) * (-1)
            data = data.astype(int)
            compl_data = [[] for _ in range(P)]
            for x, sched in Processed.items():
                tasktype = x[0]
                pr = sched[0]
                s_time = sched[1]
                e_time = s_time + Task(x).durations[env.cluster.node_types[pr]]
                data[pr, s_time:e_time] = tasktype
                if tasktype == 0:
                    compl_data[pr].insert(0, (x[1]))
                elif tasktype == 1:
                    compl_data[pr].insert(0, (x[1], x[2]))
                elif tasktype == 2:
                    compl_data[pr].insert(0, (x[1], x[2]))
                else:
                    compl_data[pr].insert(0, (x[1], x[2], x[3]))

            return data, compl_data

        def avg(a, b):
            return (a + b) / 2.0

        P = self.p
        data, compl_data = get_data(self)
        if flip:
            data = data[-1::-1, :]
            compl_data = compl_data[-1::-1]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_aspect(1)

        for y, row in enumerate(data):
            # for x, col in enumerate(row):
            x = 0
            i = 0
            indices_in_row = compl_data[y]
            while x < len(row):
                col = row[x]
                if col != -1:
                    shift = Task([col]).durations[self.cluster.node_types[y]]
                    indices = indices_in_row[i]
                else:
                    x = x + 1
                    continue
                x1 = [x, x + shift]
                y1 = np.array([y, y])
                y2 = y1 + 1
                if col == 0:
                    plt.fill_between(x1, y1, y2=y2, facecolor='green', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), 'C({})'.format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)

                if col == 1:
                    plt.fill_between(x1, y1, y2=y2, facecolor='red', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "S{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                if col == 2:
                    plt.fill_between(x1, y1, y2=y2, facecolor='orange', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "T{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                if col == 3:
                    plt.fill_between(x1, y1, y2=y2, facecolor='yellow', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "G{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                x = x + shift
                i = i + 1

        plt.ylim(P, 0)
        plt.xlim(-1e-3, data.shape[1] + 1e-3)
        plt.xticks(fontsize=50)
        if fig_file != None:
            plt.savefig(fig_file)
        return


if __name__ == "__main__":
    import torch
    from env import CholeskyTaskGraph
    import networkx as nx
    from torch_geometric.utils.convert import to_networkx

    import pydot
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_pydot import graphviz_layout
    import numpy as np

    from model import *
    pass
