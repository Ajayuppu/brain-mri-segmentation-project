"""Loss conditions for the investigation.
bce            : baseline, plain BCEWithLogits
bce_dice       : proposed, BCE + lambda * soft Dice (batch-pooled)
bce_dice_pos   : diagnostic, soft Dice computed only over positive slices in the batch
weighted_bce   : diagnostic simpler alternative, BCE with pos_weight (fixed from TRAIN foreground ratio or set)
"""
import torch
import torch.nn.functional as F


def soft_dice_loss(logits, target, eps=1.0, positive_only=False):
    p = torch.sigmoid(logits)
    if positive_only:
        keep = target.flatten(1).sum(1) > 0
        if keep.sum() == 0:
            return logits.sum() * 0.0
        p, target = p[keep], target[keep]
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)


def make_loss(name, dice_weight=1.0, pos_weight=None):
    def fn(logits, target):
        if name == "weighted_bce":
            pw = torch.tensor([pos_weight], device=logits.device)
            return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        if name == "bce":
            return bce
        if name == "bce_dice":
            return bce + dice_weight * soft_dice_loss(logits, target)
        if name == "bce_dice_pos":
            return bce + dice_weight * soft_dice_loss(logits, target, positive_only=True)
        raise ValueError(name)
    return fn
