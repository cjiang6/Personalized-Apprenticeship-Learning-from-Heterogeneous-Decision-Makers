import torch
import torch.nn.functional as F

class AlphaLoss(torch.nn.Module):
    """
    Class structure for computing alpha divergence between two distributions
    """

    def __init__(self, alpha=None):
        super(AlphaLoss, self).__init__()

        self.alpha = alpha

    def forward(self, p, q):
        """
        Computes alpha divergence between true distribution p and approximate distribution q
        :param p: True Label
        :param q: Approximate Dist
        :param alpha: tunable alpha divergence parameter
        :return:
        """
        if self.alpha == 1:
            return torch.log(p * (p / q)).sum()
        else:
            return 1 / (self.alpha - 1) * torch.log((p ** self.alpha * q ** (1 - self.alpha)).sum())
            # return 1 / (alpha - 1) * log((P * (Q / P) ** (1 - alpha)).sum())
