import argparse
from pathlib import Path
# to use
# def main():
#     # args and logger
#     args = parse_args()
#     result_logger = Logger(args) # rele does nothing tbh

def parse_args():
    parser = argparse.ArgumentParser("Scheduling Environment Dual-Bayesian Networks")
    # Environment
    parser.add_argument("--num-schedules", type=int, default=3, help="number of schedules")

    parser.add_argument('--terminal-usage', type=bool, default=False, help='Are you using the terminal?')
    parser.add_argument('--embedding-lr', type=float, default=.1, help='learning rate for embedding')
    parser.add_argument('--test-embedding-lr', type=float, default=.5, help='test learning rate for embedding')
    parser.add_argument('--network-lr', type=float, default=0.0001, help='learning rate for network')
    parser.add_argument('--no-share-params', action='store_true', default=False, help="no share parameters between value and policy networks")
    # training setting and length
    parser.add_argument("--LSTM-rollout", type=int, default=5, help="rollout distance")
    parser.add_argument("--sc2-LSTM-rollout", type=int, default=10, help="rollout distance")
    parser.add_argument('--gamma', type=float, default=.9, help='discount factor')
    parser.add_argument('--sc2-gamma', type=float, default=.995, help='discount factor')
    parser.add_argument('--stop-condition-min', type=int, default=300000, help='if algorithm has converged, and this many passes, STOP!')
    parser.add_argument('--sc2-embedding-lr', type=float, default=.1, help='test learning rate for embedding')
    parser.add_argument('--sc2-test-embedding-lr', type=float, default=.05, help='test learning rate for embedding')
    parser.add_argument('--sc2-network-lr', type=float, default=0.0001, help='learning rate for network')
    parser.add_argument('--sc2-max-training-iterations', type=int, default=1500, help='max training iterations')
    # parser.add_argument('--num-frames', type=int, default=10e6, help='number of frames to train (default: 10e6)')
    # parser.add_argument('--num-games-per-update', type=int, default=1)
    # parser.add_argument('--parallel', action='store_true', default=False)
    # parser.add_argument('--num-processes', type=int, default=2, help='how many training CPU processes to use')
    # # vis & log
    # parser.add_argument("--exp-name", type=str, default="", help="name of the experiment")
    # parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--home-dir", type=str, default="/home/ghost/PycharmProjects/scheduling_environment")
    # parser.add_argument("--load", action='store_true', default=False, help="load mddel from file")
    # # misc
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    args = parser.parse_args()
    # if args.exp_name == "":
    #     args.exp_name = args.map
    #     if args.communication:
    #         args.exp_name = args.exp_name + "_comm"
    #     if args.method != "PPO":
    #         args.exp_name = args.exp_name + "_" + args.method

    # flow2path = dirname(os.path.abspath(__file__))
    # if args.save_dir == "":
    #     args.save_dir = os.path.join(flow2path, "results")
    # if args.load_dir == "":
    #     args.load_dir = os.path.join(flow2path, "results")
    #
    # args.checkpoint_dir = os.path.join(args.load_dir, args.exp_name)
    # args.reward_dir = os.path.join(args.save_dir, "reward")
    #
    # if not os.path.isdir(args.save_dir):
    #     os.mkdir(args.save_dir)
    # if not os.path.isdir(args.reward_dir):
    #     os.mkdir(args.reward_dir)
    # if not os.path.isdir(args.checkpoint_dir):
    #     os.mkdir(args.checkpoint_dir)
    #
    # if args.rb_load_iteration == 0:
    #     args.rb_load_iteration = args.load_iteration

    # torch.manual_seed(args.seed)

    return args


class Logger():
    def __init__(self):

        self.args = parse_args()
        self.LSTM_rollout = self.args.LSTM_rollout
        self.num_schedules = self.args.num_schedules
        self.network_lr = self.args.network_lr
        self.embedding_lr = self.args.embedding_lr
        self.terminal_usage = self.args.terminal_usage
        self.gamma = self.args.gamma
        self.stop_condition_min = self.args.stop_condition_min
        self.test_embedding_lr = self.args.test_embedding_lr
        self.home_dir = self.args.home_dir
        self.sc2_network_lr = self.args.sc2_network_lr
        self.sc2_embedding_lr = self.args.sc2_embedding_lr
        self.sc2_max_training_iterations = self.args.sc2_max_training_iterations
        self.sc2_rollout = self.args.sc2_LSTM_rollout
        self.sc2_test_embedding_lr = self.args.sc2_test_embedding_lr
        self.sc2_gamma = self.args.sc2_gamma
