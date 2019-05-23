# Created by Ghost on 2/21/19
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
# def log_gaussian_prob(x, mu, sigma, log_sigma=False):
#     if not log_sigma:
#         element_wise_log_prob = -0.5*torch.Tensor([np.log(2*np.pi)]).to(mu.device) - torch.log(sigma) - 0.5*(x-mu)**2 / sigma**2
#     else:
#         element_wise_log_prob = -0.5*torch.Tensor([np.log(2*np.pi)]).to(mu.device) - F.softplus(sigma) - 0.5*(x-mu)**2 / F.softplus(sigma)**2
#     return element_wise_log_prob.sum()
#
# class GaussianLinear(nn.Module):
#     def __init__(self, in_dim, out_dim, stddev_prior, bias=False):
#         super(GaussianLinear, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.stddev_prior = stddev_prior
#         self.w_mu = nn.Parameter(torch.Tensor(in_dim, out_dim).normal_(0, stddev_prior))
#         self.w_rho = nn.Parameter(torch.Tensor(in_dim, out_dim).normal_(0, stddev_prior))
#         self.b_mu = nn.Parameter(torch.Tensor(out_dim).normal_(0, stddev_prior)) if bias else None
#         self.b_rho = nn.Parameter(torch.Tensor(out_dim).normal_(0, stddev_prior)) if bias else None
#         self.bias = bias
#         self.q_w = 0.
#         self.p_w = 0.
#
#     def forward(self, x, test=False):
#         if test:
#             w = self.w_mu
#             b = self.b_mu if self.bias else None
#         else:
#             device = self.w_mu.device
#             w_stddev = F.softplus(self.w_rho)
#             b_stddev = F.softplus(self.b_rho) if self.bias else None
#             w = self.w_mu + w_stddev * torch.Tensor(self.in_dim, self.out_dim).to(device).normal_(0,self.stddev_prior)
#             b = self.b_mu + b_stddev * torch.Tensor(self.out_dim).to(device).normal_(0,self.stddev_prior) if self.bias else None
#             self.q_w = log_gaussian_prob(w, self.w_mu, self.w_rho, log_sigma=True)
#             self.p_w = log_gaussian_prob(w, torch.zeros_like(self.w_mu, device=device), self.stddev_prior*torch.ones_like(w_stddev, device=device))
#             if self.bias:
#                 self.q_w += log_gaussian_prob(b, self.b_mu, self.b_rho, log_sigma=True)
#                 self.p_w += log_gaussian_prob(b, torch.zeros_like(self.b_mu, device=device), self.stddev_prior*torch.ones_like(b_stddev, device=device))
#         output = w.mul(x)
#         return output
#
#     def get_pw(self):
#         return self.p_w
#
#     def get_qw(self):
#         return self.q_w
#
#     def get_w_forward(self, test=False):
#         if test:
#             return self.w_mu
#         w_stddev = F.softplus(self.w_rho)
#         w = self.w_mu + w_stddev * torch.Tensor(self.in_dim, self.out_dim).normal_(0, self.stddev_prior)
#         self.q_w = log_gaussian_prob(w, self.w_mu, self.w_rho, log_sigma=True)
#         self.p_w = log_gaussian_prob(w, torch.zeros_like(self.w_mu),
#                                      self.stddev_prior * torch.ones_like(w_stddev))
#         return w
# class Gaussian_ProLoNet(nn.Module):
#     def __init__(self,
#                  input_dim,
#                  weights,
#                  comparators,
#                  leaves,
#                  selectors=None,
#                  output_dim=None,
#                  bayesian_embedding_dim=None,
#                  alpha=1.0,
#                  freeze_alpha=False,
#                  is_value=False,
#                  use_gpu=False,
#                  vectorized=True,
#                  stddev_prior=1):
#         super(Gaussian_ProLoNet, self).__init__()
#         """
#         Initialize the ProLoNet, taking in premade weights for inputs to comparators and sigmoids
#         Alternatively, pass in None to everything except for input_dim and output_dim, and you will get a randomly
#         initialized tree. If you pass an int to leaves, it must be 2**N so that we can build a balanced tree
#         :param input_dim: int. always required for input dimensionality
#         :param weights: None or a list of lists, where each sub-list is a weight vector for each node
#         :param comparators: None or a list of lists, where each sub-list is a comparator vector for each node
#         :param leaves: None, int, or truple of [[left turn indices], [right turn indices], [final_probs]]. If int, must be 2**N
#         :param output_dim: None or int, must be an int if weights and comparators are None
#         :param alpha: int. Strictness of the tree, default 1
#         :param is_value: if False, outputs are passed through a Softmax final layer. Default: False
#         :param use_gpu: is this a GPU-enabled network? Default: False
#         :param vectorized: Use a vectorized comparator? Default: True
#         """
#         self.use_gpu = use_gpu
#         self.vectorized = vectorized
#         self.leaf_init_information = leaves
#         self.bayesian_embedding_dim = bayesian_embedding_dim
#         self.freeze_alpha = freeze_alpha
#         self.stddev_prior = stddev_prior
#
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.layers = None
#         self.comparators = None
#         self.bayesian_embedding = None
#         self.selector = None
#
#         self.init_bayesian_embedding()
#         self.init_comparators(comparators)
#         self.init_weights(weights)
#         self.init_alpha(alpha)
#         if self.vectorized:
#             self.init_selector(selectors, weights)
#         self.init_paths()
#         self.init_leaves()
#         self.added_levels = nn.Sequential()
#
#         self.sig = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=-1)
#         self.is_value = is_value
#
#     def init_bayesian_embedding(self):
#         if self.bayesian_embedding_dim is not None:
#             self.input_dim += self.bayesian_embedding_dim
#             bayes_embed = torch.Tensor(np.random.normal(0, 1, self.bayesian_embedding_dim))
#             bayes_embed.requires_grad = True
#             self.bayesian_embedding = nn.Parameter(bayes_embed)
#
#     def init_comparators(self, comparators):
#         if comparators is None:
#             comparators = []
#             if type(self.leaf_init_information) is int:
#                 depth = int(np.floor(np.log2(self.leaf_init_information)))
#             else:
#                 depth = 4
#             for level in range(depth):
#                 for node in range(2**level):
#                     comparators.append(np.random.normal(0, self.stddev_prior, self.input_dim))
#         self.comparators = GaussianLinear(in_dim=len(comparators), out_dim=self.input_dim, stddev_prior=self.stddev_prior)
#         # new_comps = torch.Tensor(comparators)
#         # new_comps.requires_grad = True
#         # if self.use_gpu:
#         #     new_comps = new_comps.cuda()
#         # self.comparators = nn.Parameter(new_comps)
#
#     def init_weights(self, weights):
#         if weights is None:
#             weights = []
#             if type(self.leaf_init_information) is int:
#                 depth = int(np.floor(np.log2(self.leaf_init_information)))
#             else:
#                 depth = 4
#             for level in range(depth):
#                 for node in range(2**level):
#                     weights.append(np.random.normal(0, self.stddev_prior, self.input_dim))
#         self.layers = GaussianLinear(in_dim=len(weights), out_dim=self.input_dim, stddev_prior=self.stddev_prior)
#         # new_weights = torch.Tensor(weights)
#         # new_weights.requires_grad = True
#         # if self.use_gpu:
#         #     new_weights = new_weights.cuda()
#         # self.layers = nn.Parameter(new_weights)
#
#     def init_alpha(self, alpha):
#         self.alpha = torch.Tensor([alpha])
#         if self.use_gpu:
#             self.alpha = self.alpha.cuda()
#         if not self.freeze_alpha:
#             self.alpha.requires_grad = True
#             self.alpha = nn.Parameter(self.alpha)
#
#     def init_selector(self, selector, weights):
#         if selector is None:
#             if weights is None:
#                 selector = np.ones(self.layers.get_w_forward(True).size())*(1.0/self.input_dim)
#             else:
#                 selector = []
#                 for layer in self.layers:
#                     new_sel = np.zeros(layer.size())
#                     max_ind = torch.argmax(torch.abs(layer)).item()
#                     new_sel[max_ind] = 1
#                     selector.append(new_sel)
#         selector = torch.Tensor(selector)
#         selector.requires_grad = True
#         if self.use_gpu:
#             selector = selector.cuda()
#         self.selector = nn.Parameter(selector)
#
#     def init_paths(self):
#         if type(self.leaf_init_information) is list:
#             left_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)))
#             right_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)))
#             for n in range(0, len(self.leaf_init_information)):
#                 for i in self.leaf_init_information[n][0]:
#                     left_branches[i][n] = 1.0
#                 for j in self.leaf_init_information[n][1]:
#                     right_branches[j][n] = 1.0
#         else:
#             if type(self.leaf_init_information) is int:
#                 depth = int(np.floor(np.log2(self.leaf_init_information)))
#             elif self.leaf_init_information is None:
#                 depth = 4
#             left_branches = torch.zeros((2 ** depth - 1, 2 ** depth))
#             for n in range(0, depth):
#                 row = 2 ** n - 1
#                 for i in range(0, 2 ** depth):
#                     col = 2 ** (depth - n) * i
#                     end_col = col + 2 ** (depth - 1 - n)
#                     if row + i >= len(left_branches) or end_col >= len(left_branches[row]):
#                         break
#                     left_branches[row + i, col:end_col] = 1.0
#             right_branches = torch.zeros((2 ** depth - 1, 2 ** depth))
#             left_turns = np.where(left_branches == 1)
#             for row in np.unique(left_turns[0]):
#                 cols = left_turns[1][left_turns[0] == row]
#                 start_pos = cols[-1] + 1
#                 end_pos = start_pos + len(cols)
#                 right_branches[row, start_pos:end_pos] = 1.0
#         left_branches.requires_grad = False
#         right_branches.requires_grad = False
#         if self.use_gpu:
#             left_branches = left_branches.cuda()
#             right_branches = right_branches.cuda()
#         self.left_path_sigs = left_branches
#         self.right_path_sigs = right_branches
#
#     def init_leaves(self):
#         if type(self.leaf_init_information) is list:
#             new_leaves = [leaf[-1] for leaf in self.leaf_init_information]
#         else:
#             new_leaves = []
#             if type(self.leaf_init_information) is int:
#                 depth = int(np.floor(np.log2(self.leaf_init_information)))
#             else:
#                 depth = 4
#
#             last_level = np.arange(2**(depth-1)-1, 2**depth-1)
#             going_left = True
#             leaf_index = 0
#             self.leaf_init_information = []
#             for level in range(2**depth):
#                 curr_node = last_level[leaf_index]
#                 turn_left = going_left
#                 left_path = []
#                 right_path = []
#                 while curr_node >= 0:
#                     if turn_left:
#                         left_path.append(int(curr_node))
#                     else:
#                         right_path.append(int(curr_node))
#                     prev_node = np.ceil(curr_node / 2) - 1
#                     if curr_node // 2 > prev_node:
#                         turn_left = False
#                     else:
#                         turn_left = True
#                     curr_node = prev_node
#                 if going_left:
#                     going_left = False
#                 else:
#                     going_left = True
#                     leaf_index += 1
#                 new_probs = np.random.normal(0, self.stddev_prior, self.output_dim)  # *(1.0/self.output_dim)
#                 self.leaf_init_information.append([sorted(left_path), sorted(right_path), new_probs])
#                 new_leaves.append(new_probs)
#         self.action_probs = GaussianLinear(in_dim=len(new_leaves), out_dim=self.output_dim, stddev_prior=self.stddev_prior)
#         # labels = torch.Tensor(new_leaves)
#         # if self.use_gpu:
#         #     labels = labels.cuda()
#         # labels.requires_grad = True
#         # self.action_probs = nn.Parameter(labels)
#
#     def forward(self, input_data, embedding_list=None, test=False):
#         if self.bayesian_embedding is not None:
#             if embedding_list is not None:
#                 input_temp = [torch.cat((input_data[0], self.bayesian_embedding))]
#                 for e_ind, embedding in enumerate(embedding_list):
#                     embedding = torch.Tensor(embedding)
#                     if self.use_gpu:
#                         embedding = embedding.cuda()
#                     input_temp.append(torch.cat((input_data[e_ind+1], embedding)))
#                 input_data = torch.stack(input_temp)
#             else:
#                 input_data = torch.cat((input_data, self.bayesian_embedding.expand(input_data.size(0),
#                                                                                    *self.bayesian_embedding.size())), dim=1)
#
#         input_data = input_data.t().expand(self.layers.get_w_forward(True).size(0), *input_data.t().size())
#
#         input_data = input_data.permute(2, 0, 1)
#         comp = self.layers(input_data, test)
#         comparator = self.comparators.get_w_forward(test)
#         comparator = comparator.expand(input_data.size(0), *comparator.size())
#         comp = comp.sub(comparator)
#         comp = comp.mul(self.alpha)
#         sig_vals = self.sig(comp)
#
#         s_temp_main = self.selector
#         selector_subber = self.selector.detach().clone()
#         selector_divver = self.selector.detach().clone()
#         selector_subber[np.arange(0, len(selector_subber)), selector_subber.max(dim=1)[1]] = 0
#         selector_divver[selector_divver == 0] = 1
#         s_temp_main = s_temp_main.sub(selector_subber)
#         s_temp_main = s_temp_main.div(selector_divver)
#
#         s_temp_main = s_temp_main.expand(input_data.size(0), *self.selector.size())
#
#         sig_vals = sig_vals.mul(s_temp_main)
#         sig_vals = sig_vals.sum(dim=2)
#
#         sig_vals = sig_vals.view(input_data.size(0), -1)
#
#         if not self.use_gpu:
#             one_minus_sig = torch.ones(sig_vals.size()).sub(sig_vals)
#         else:
#             one_minus_sig = torch.ones(sig_vals.size()).cuda().sub(sig_vals)
#
#         left_path_probs = self.left_path_sigs.t()
#         right_path_probs = self.right_path_sigs.t()
#         left_path_probs = left_path_probs.expand(input_data.size(0), *left_path_probs.size()) * sig_vals.unsqueeze(1)
#         right_path_probs = right_path_probs.expand(input_data.size(0), *right_path_probs.size()) * one_minus_sig.unsqueeze(1)
#         left_path_probs = left_path_probs.permute(0, 2, 1)
#         right_path_probs = right_path_probs.permute(0, 2, 1)
#
#         left_filler = torch.zeros(self.left_path_sigs.size())
#         left_filler[self.left_path_sigs == 0] = 1
#         right_filler = torch.zeros(self.right_path_sigs.size())
#         if self.use_gpu:
#             left_filler = left_filler.cuda()
#             right_filler = right_filler.cuda()
#         right_filler[self.right_path_sigs == 0] = 1
#
#         left_path_probs = left_path_probs.add(left_filler)
#         right_path_probs = right_path_probs.add(right_filler)
#
#         probs = torch.cat((left_path_probs, right_path_probs), dim=1)
#         probs = probs.prod(dim=1)
#
#         actions = probs.mm(self.action_probs.get_w_forward(test))
#
#         if not self.is_value:
#             return self.softmax(actions)
#         else:
#             return actions
#
#     def forward_samples(self, x, y, nb_samples=3):
#         total_qw, total_pw, total_log_likelihood = 0., 0., 0.
#         for _ in range(nb_samples):
#             output = self.forward(x)
#             total_qw += self.get_qw()
#             total_pw += self.get_pw()
#             y = y.view(len(y), -1)
#             y_onehot = torch.Tensor(len(y), self.output_dim).to(x.device)
#             y_onehot.zero_()
#             y_onehot.scatter_(1, y.long(), 1)
#             total_log_likelihood += log_gaussian_prob(y_onehot, output, self.stddev_prior*torch.ones_like(y_onehot, device=y_onehot.device))
#         return total_qw / nb_samples, total_pw / nb_samples, total_log_likelihood / nb_samples
#
#     def get_pw(self):
#         return self.layers.p_w + self.comparators.p_w + self.action_probs.p_w
#
#     def get_qw(self):
#         return self.layers.q_w + self.comparators.q_w + self.action_probs.q_w
#
#     def get_bayesian_embedding(self):
#         if self.bayesian_embedding is not None:
#             return self.bayesian_embedding.data.cpu().numpy()
#         else:
#             return None
#
#     def set_bayesian_embedding(self, embedding_data):
#         if self.bayesian_embedding is not None:
#             new_embed = torch.Tensor(embedding_data)
#             if self.use_gpu:
#                 new_embed = new_embed.cuda()
#             self.bayesian_embedding.data = new_embed
#         else:
#             raise AttributeError("Network was not initialized with a Bayesian embedding")
#
#     def reset_bayesian_embedding(self):
#         self.init_bayesian_embedding()

class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """

    def __init__(self, input_features, output_features, prior_var=6):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        super().__init__()
        # set dim
        self.input_features = input_features
        self.output_features = output_features

        # initialize weight params
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-0.6, 0.6))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-6, -6))

        # initialize weight samples
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon
        # initialize prior distribution
        self.prior = torch.distributions.Normal(0, prior_var)

    def forward(self, input):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # record prior
        w_log_prior = self.prior.log_prob(self.w)
        self.log_prior = torch.sum(w_log_prior)

        # record variational_posterior
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum()
        return F.linear(input, self.w)

    def get_w(self):
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        self.log_prior = torch.sum(w_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum()
        return self.w


class Gaussian_ProLoNet(nn.Module):
    def __init__(self,
                 input_dim,
                 weights,
                 comparators,
                 leaves,
                 selectors=None,
                 output_dim=None,
                 bayesian_embedding_dim=None,
                 alpha=1.0,
                 freeze_alpha=False,
                 is_value=False,
                 use_gpu=False,
                 vectorized=True,
                 prior_var=1,
                 noise_tol=0.1,
                 num_batches=1):
        super(Gaussian_ProLoNet, self).__init__()
        """
        Initialize the ProLoNet, taking in premade weights for inputs to comparators and sigmoids
        Alternatively, pass in None to everything except for input_dim and output_dim, and you will get a randomly
        initialized tree. If you pass an int to leaves, it must be 2**N so that we can build a balanced tree
        :param input_dim: int. always required for input dimensionality
        :param weights: None or a list of lists, where each sub-list is a weight vector for each node
        :param comparators: None or a list of lists, where each sub-list is a comparator vector for each node
        :param leaves: None, int, or truple of [[left turn indices], [right turn indices], [final_probs]]. If int, must be 2**N
        :param output_dim: None or int, must be an int if weights and comparators are None
        :param alpha: int. Strictness of the tree, default 1
        :param is_value: if False, outputs are passed through a Softmax final layer. Default: False
        :param use_gpu: is this a GPU-enabled network? Default: False
        :param vectorized: Use a vectorized comparator? Default: True
        """
        self.use_gpu = use_gpu
        self.vectorized = vectorized
        self.leaf_init_information = leaves
        self.bayesian_embedding_dim = bayesian_embedding_dim
        self.freeze_alpha = freeze_alpha
        self.prior_var = prior_var
        self.noise_tol = noise_tol
        self.num_batches = num_batches

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = None
        self.comparators = None
        self.bayesian_embedding = None
        self.selector = None

        self.init_bayesian_embedding()
        self.init_comparators(comparators)
        self.init_weights(weights)
        self.init_alpha(alpha)
        if self.vectorized:
            self.init_selector(selectors, weights)
        self.init_paths()
        self.init_leaves()
        self.added_levels = nn.Sequential()

        self.sig = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)
        self.is_value = is_value

    def init_bayesian_embedding(self):
        if self.bayesian_embedding_dim is not None:
            self.input_dim += self.bayesian_embedding_dim
            bayes_embed = torch.Tensor(np.random.normal(0, 1, self.bayesian_embedding_dim))
            bayes_embed.requires_grad = True
            self.bayesian_embedding = nn.Parameter(bayes_embed)

    def init_comparators(self, comparators):
        if comparators is None:
            comparators = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            for level in range(depth):
                for node in range(2**level):
                    comparators.append(np.random.normal(0, 1, self.input_dim))
        self.comparators = Linear_BBB(len(comparators), self.input_dim, prior_var=self.prior_var)
        # new_comps = torch.Tensor(comparators)
        # new_comps.requires_grad = True
        # if self.use_gpu:
        #     new_comps = new_comps.cuda()
        # self.comparators = nn.Parameter(new_comps)

    def init_weights(self, weights):
        if weights is None:
            weights = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            for level in range(depth):
                for node in range(2**level):
                    weights.append(np.random.normal(0, 1, self.input_dim))
        self.layers = Linear_BBB(len(weights), self.input_dim, prior_var=self.prior_var)
        # new_weights = torch.Tensor(weights)
        # new_weights.requires_grad = True
        # if self.use_gpu:
        #     new_weights = new_weights.cuda()
        # self.layers = nn.Parameter(new_weights)

    def init_alpha(self, alpha):
        self.alpha = torch.Tensor([alpha])
        if self.use_gpu:
            self.alpha = self.alpha.cuda()
        if not self.freeze_alpha:
            self.alpha.requires_grad = True
            self.alpha = nn.Parameter(self.alpha)

    def init_selector(self, selector, weights):
        if selector is None:
            if weights is None:
                selector = np.ones(self.layers.get_w().size())*(1.0/self.input_dim)
            else:
                selector = []
                for layer in self.layers:
                    new_sel = np.zeros(layer.size())
                    max_ind = torch.argmax(torch.abs(layer)).item()
                    new_sel[max_ind] = 1
                    selector.append(new_sel)
        selector = torch.Tensor(selector)
        selector.requires_grad = True
        if self.use_gpu:
            selector = selector.cuda()
        self.selector = nn.Parameter(selector)

    def init_paths(self):
        if type(self.leaf_init_information) is list:
            left_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)))
            right_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)))
            for n in range(0, len(self.leaf_init_information)):
                for i in self.leaf_init_information[n][0]:
                    left_branches[i][n] = 1.0
                for j in self.leaf_init_information[n][1]:
                    right_branches[j][n] = 1.0
        else:
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            elif self.leaf_init_information is None:
                depth = 4
            left_branches = torch.zeros((2 ** depth - 1, 2 ** depth))
            for n in range(0, depth):
                row = 2 ** n - 1
                for i in range(0, 2 ** depth):
                    col = 2 ** (depth - n) * i
                    end_col = col + 2 ** (depth - 1 - n)
                    if row + i >= len(left_branches) or end_col >= len(left_branches[row]):
                        break
                    left_branches[row + i, col:end_col] = 1.0
            right_branches = torch.zeros((2 ** depth - 1, 2 ** depth))
            left_turns = np.where(left_branches == 1)
            for row in np.unique(left_turns[0]):
                cols = left_turns[1][left_turns[0] == row]
                start_pos = cols[-1] + 1
                end_pos = start_pos + len(cols)
                right_branches[row, start_pos:end_pos] = 1.0
        left_branches.requires_grad = False
        right_branches.requires_grad = False
        if self.use_gpu:
            left_branches = left_branches.cuda()
            right_branches = right_branches.cuda()
        self.left_path_sigs = left_branches
        self.right_path_sigs = right_branches

    def init_leaves(self):
        if type(self.leaf_init_information) is list:
            new_leaves = [leaf[-1] for leaf in self.leaf_init_information]
        else:
            new_leaves = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4

            last_level = np.arange(2**(depth-1)-1, 2**depth-1)
            going_left = True
            leaf_index = 0
            self.leaf_init_information = []
            for level in range(2**depth):
                curr_node = last_level[leaf_index]
                turn_left = going_left
                left_path = []
                right_path = []
                while curr_node >= 0:
                    if turn_left:
                        left_path.append(int(curr_node))
                    else:
                        right_path.append(int(curr_node))
                    prev_node = np.ceil(curr_node / 2) - 1
                    if curr_node // 2 > prev_node:
                        turn_left = False
                    else:
                        turn_left = True
                    curr_node = prev_node
                if going_left:
                    going_left = False
                else:
                    going_left = True
                    leaf_index += 1
                new_probs = np.random.normal(0, 1, self.output_dim)  # *(1.0/self.output_dim)
                self.leaf_init_information.append([sorted(left_path), sorted(right_path), new_probs])
                new_leaves.append(new_probs)
        self.action_probs = Linear_BBB(len(new_leaves), self.output_dim, prior_var=self.prior_var)
        # labels = torch.Tensor(new_leaves)
        # if self.use_gpu:
        #     labels = labels.cuda()
        # labels.requires_grad = True
        # self.action_probs = nn.Parameter(labels)

    def forward(self, input_data, embedding_list=None, test=False):
        if self.bayesian_embedding is not None:
            if embedding_list is not None:
                input_temp = [torch.cat((input_data[0], self.bayesian_embedding))]
                for e_ind, embedding in enumerate(embedding_list):
                    embedding = torch.Tensor(embedding)
                    if self.use_gpu:
                        embedding = embedding.cuda()
                    input_temp.append(torch.cat((input_data[e_ind+1], embedding)))
                input_data = torch.stack(input_temp)
            else:
                input_data = torch.cat((input_data, self.bayesian_embedding.expand(input_data.size(0),
                                                                                   *self.bayesian_embedding.size())), dim=1)

        input_data = input_data.t().expand(self.layers.get_w().size(0), *input_data.t().size())

        input_data = input_data.permute(2, 0, 1)
        comp = self.layers.get_w().mul(input_data)
        comparator = self.comparators.get_w()
        comparator = comparator.expand(input_data.size(0), *comparator.size())
        comp = comp.sub(comparator)
        comp = comp.mul(self.alpha)
        sig_vals = self.sig(comp)

        s_temp_main = self.selector
        selector_subber = self.selector.detach().clone()
        selector_divver = self.selector.detach().clone()
        selector_subber[np.arange(0, len(selector_subber)), selector_subber.max(dim=1)[1]] = 0
        selector_divver[selector_divver == 0] = 1
        s_temp_main = s_temp_main.sub(selector_subber)
        s_temp_main = s_temp_main.div(selector_divver)

        s_temp_main = s_temp_main.expand(input_data.size(0), *self.selector.size())

        sig_vals = sig_vals.mul(s_temp_main)
        sig_vals = sig_vals.sum(dim=2)

        sig_vals = sig_vals.view(input_data.size(0), -1)

        if not self.use_gpu:
            one_minus_sig = torch.ones(sig_vals.size()).sub(sig_vals)
        else:
            one_minus_sig = torch.ones(sig_vals.size()).cuda().sub(sig_vals)

        left_path_probs = self.left_path_sigs.t()
        right_path_probs = self.right_path_sigs.t()
        left_path_probs = left_path_probs.expand(input_data.size(0), *left_path_probs.size()) * sig_vals.unsqueeze(1)
        right_path_probs = right_path_probs.expand(input_data.size(0), *right_path_probs.size()) * one_minus_sig.unsqueeze(1)
        left_path_probs = left_path_probs.permute(0, 2, 1)
        right_path_probs = right_path_probs.permute(0, 2, 1)

        left_filler = torch.zeros(self.left_path_sigs.size())
        left_filler[self.left_path_sigs == 0] = 1
        right_filler = torch.zeros(self.right_path_sigs.size())
        if self.use_gpu:
            left_filler = left_filler.cuda()
            right_filler = right_filler.cuda()
        right_filler[self.right_path_sigs == 0] = 1

        left_path_probs = left_path_probs.add(left_filler)
        right_path_probs = right_path_probs.add(right_filler)

        probs = torch.cat((left_path_probs, right_path_probs), dim=1)
        probs = probs.prod(dim=1)

        actions = probs.mm(self.action_probs.get_w().t())

        if not self.is_value:
            return self.softmax(actions)
        else:
            return actions

    def log_prior(self):
        # calculate the log prior over all the layers
        return self.layers.log_prior + self.comparators.log_prior + self.action_probs.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.layers.log_post + self.comparators.log_post + self.action_probs.log_post

    def sample_elbo(self, input, target, samples):
        # we calculate the negative elbo, which will be our loss function
        # initialize tensors
        if self.is_value:
            outputs = torch.zeros(samples, target.shape[0])
        else:
            outputs = torch.zeros(samples, target.shape[0], self.output_dim)
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input).reshape(-1)  # make predictions
            log_priors[i] = self.log_prior()  # get log prior
            log_posts[i] = self.log_post()  # get log variational posterior
            if self.is_value:
                log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum()  # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        if self.is_value:
            log_like = log_likes.mean()
            # calculate the negative elbo (which is our loss function)
            loss = log_post - log_prior - log_like
            return loss
        else:
            log_likes = F.nll_loss(outputs.mean(0), target.reshape(-1), reduction='sum')
            loss = (log_post - log_prior) / self.num_batches + log_likes
            return loss, outputs

    def get_bayesian_embedding(self):
        if self.bayesian_embedding is not None:
            return self.bayesian_embedding.data.cpu().numpy()
        else:
            return None

    def set_bayesian_embedding(self, embedding_data):
        if self.bayesian_embedding is not None:
            new_embed = torch.Tensor(embedding_data)
            if self.use_gpu:
                new_embed = new_embed.cuda()
            self.bayesian_embedding.data._fill(new_embed)
        else:
            raise AttributeError("Network was not initialized with a Bayesian embedding")

    def reset_bayesian_embedding(self):
        self.init_bayesian_embedding()
