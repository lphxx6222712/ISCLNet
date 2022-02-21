import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.autograd import Function


def loss_builder(loss_type):
    weight_1 = torch.Tensor([0.01, 20, 20, 20])  # weighted CE
    criterion_1 = nn.NLLLoss(weight_1).cuda()
    criterion_2 = DiceLoss().cuda()
    criterion_3 = nn.BCELoss().cuda()

    criterion = [criterion_1, criterion_2, criterion_3]
    return criterion


class DiceLoss(nn.Module):
    def __init__(self, class_num=3, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        input = torch.exp(input)
        self.smooth = 0.
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice / (self.class_num - 1)
        return dice_loss


class EL_DiceLoss(nn.Module):
    def __init__(self, class_num=3, smooth=1, gamma=0.5):
        super(EL_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.gamma = gamma

    def forward(self, input, target):
        input = torch.exp(input)
        self.smooth = 0.
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += (-torch.log(dice)) ** self.gamma
        dice_loss = Dice / (self.class_num - 1)
        return dice_loss


class focal_loss(nn.Module):
    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction_mode)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


class DenseCRFLossFunction(Function):

    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape

        ROIs = ROIs.unsqueeze_(1).repeat(1, ctx.K, 1, 1)
        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs

        densecrf_loss = 0.0
        images = images.numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        # bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        densecrf_loss -= np.dot(segmentations, AS)

        # averaged by the number of images
        densecrf_loss /= ctx.N

        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2 * grad_output * torch.from_numpy(ctx.AS) / ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None


class DenseCRFLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseCRFLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor

    def forward(self, images, segmentations, ROIs):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images, scale_factor=self.scale_factor)
        scaled_segs = F.interpolate(segmentations, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1), scale_factor=self.scale_factor).squeeze(1)
        return self.weight * DenseCRFLossFunction.apply(
            scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy * self.scale_factor, scaled_ROIs)
