from env import RDAGEnv
from a2c import A2C
from a2c import *
from model import Net, SimpleNet, SimpleNet2, ResNetG, SimpleNetMax, ModelHeterogene
from log_utils import name_dir_logger, name_dir
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import numpy as np
parser = argparse.ArgumentParser()
# Training settings

parser.add_argument('--model_path', type=str, default='none', help='path to load model')
parser.add_argument('--num_env_steps', type=int, default=10 ** 4, help='num env steps')
parser.add_argument('--num_processes', type=int, default=1, help='num proc')
parser.add_argument('--lr', type=float, default=10 ** -2, help='learning rate')
parser.add_argument('--eps', type=float, default=10 ** -1, help='Random seed.')
parser.add_argument('--optimizer', type=str, default='rms', help='sgd or adam or rms')
parser.add_argument('--scheduler', type=str, default='lambda', help='lambda or cyclic')
parser.add_argument('--step_up', type=float, default=100, help='step_size_up for cyclic scheduler')
parser.add_argument('--sched_ratio', type=float, default=10, help='lr ratio for cyclic scheduler')
parser.add_argument('--entropy_coef', type=float, default=0.002, help='entropy loss weight')
parser.add_argument('--gamma', type=float, default=1, help='inflation')
parser.add_argument('--loss_ratio', type=float, default=0.5, help='value loss weight')
parser.add_argument('--trajectory_length', type=int, default=40, help='batch size')
parser.add_argument('--log_interval', type=int, default=10, help='evaluate every log_interval steps')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--agent', type=str, default='A2C', help='A2C')
parser.add_argument("--result_name", type=str, default="results.csv", help="filename where results are stored")

# model settings
parser.add_argument('--input_dim', type=int, default=19, help='input dim')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dim')
parser.add_argument('--ngcn', type=int, default=0, help='number of gcn')
parser.add_argument('--nmlp', type=int, default=1, help='number of mlp to compute probs')
parser.add_argument('--nmlp_value', type=int, default=1, help='number of mlp to compute v')
parser.add_argument('--res', action='store_true', default=False, help='with residual connexion')
parser.add_argument('--withbn', action='store_true', default=False, help='with batch norm')

# env settings
parser.add_argument('--task_nodes', type=int, default=50, help='number of tasks')
parser.add_argument('--processor_nodes', type=int, default=10, help='number of processors')
parser.add_argument('--window', type=int, default=0, help='window')
parser.add_argument('--env_type', type=str, default='RouE', help='')
parser.add_argument('--seed_env', type=int, default=42, help='Random seed env ')
parser.add_argument('--edges', type=float, default=0.07, help='Number of edges or percentage of maximal edges for RouE case')
parser.add_argument('--as_density', default=True, action='store_true', help="Whether to treat edge count as a density metric or fixed num")
parser.add_argument('--noise', type=float, default=0, help='noise')


args = parser.parse_args()
config_enhanced = vars(args)

writer = SummaryWriter('runs')

print("Current config_enhanced is:")
print(config_enhanced)

main_path = "HPC"


env = RDAGEnv(args)
# env.reset()
# done = False
# while (not done):
#     _, _, done, _ = env.step(0)

model = ModelHeterogene(input_dim=args.input_dim,
                        hidden_dim=args.hidden_dim,
                        ngcn=args.ngcn,
                        nmlp=args.nmlp,
                        nmlp_value=args.nmlp_value,
                        res=args.res,
                        withbn=args.withbn)

agent = A2C(config_enhanced, env, model=model, writer=writer)

best_perf, _ = agent.training_batch()
