import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class CrossEntropy3d(nn.Module):

    def __init__(self, ignore_label=255):
        super(CrossEntropy3d, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, d, h, w)
                target:(n, d, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 5
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(2))
        assert predict.size(4) == target.size(3), "{0} vs {1} ".format(predict.size(4), target.size(3))
        n, c, d, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous()
        predict = predict[target_mask.view(n, d, h, w, 1).repeat(1, 1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight)
        return loss


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def dice_loss(score, target):
    assert score.shape == target.shape
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


class DiceLossWithSoftmax3D(nn.Module):
    def __init__(self):
        super(DiceLossWithSoftmax3D, self).__init__()
        self.eps = 1e-6

    def forward(self, input, target):
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 5:
            raise ValueError("Invalid input shape, we expect BxNxDxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-3:] == target.shape[-3:]:
            raise ValueError("input and target shapes must be the same. Got: {} {}"
                             .format(input.shape, target.shape))
        input_soft = torch.softmax(input, dim=1)
        # compute the actual dice score
        dims = (2, 3, 4)
        intersection = torch.sum(input_soft * target, dims)
        cardinality = torch.sum(torch.sum(input_soft * input_soft, dims) + torch.sum(target * target, dims), dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(torch.tensor(1.) - dice_score)


