import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Compute gradient of the Lovász extension w.r.t sorted errors.
    """

    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # Cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax loss for semantic segmentation.

    Args:
        ignore_index: class index to ignore
        reduction: 'mean' or 'sum'
    """

    def __init__(self, ignore_index=-100, reduction='mean'):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, C, H, W) - raw predictions
            labels: (B, H, W) - ground truth labels

        Returns:
            loss: scalar
        """

        B, C, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        labels = labels.reshape(-1)  # (B*H*W,)

        # Filter ignored labels
        valid = labels != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0

        logits = logits[valid]
        labels = labels[valid]

        probs = F.softmax(logits, dim=1)

        losses = []
        for c in range(C):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue

            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]

            grad = lovasz_grad(fg_sorted)
            loss = torch.dot(errors_sorted, grad)
            losses.append(loss)

        if len(losses) == 0:
            return logits.sum() * 0.0

        loss = torch.stack(losses)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy + Lovász-Softmax Loss.

    Args:
        weight: class weights for Cross-Entropy (optional)
        lovasz_weight: weight coefficient for Lovász loss (default: 1.0)
        ignore_index: class to ignore
    """

    def __init__(self, weight=None, lovasz_weight=1.0, ignore_index=-100):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.lovasz_loss = LovaszSoftmaxLoss(ignore_index=ignore_index)
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, C, H, W) - predictions
            labels: (B, H, W) - ground truth

        Returns:
            loss: scalar
        """

        ce = self.ce_loss(logits, labels)
        lovasz = self.lovasz_loss(logits, labels)

        return ce + self.lovasz_weight * lovasz


class FocalLoss(nn.Module):
    """
    Focal Loss для сегментації з дисбалансом класів.

    Args:
        alpha: class weights (list or float)
        gamma: focusing parameter (default: 2.0)
        num_classes: number of classes
        reduction: 'mean' or 'sum'
    """

    def __init__(self, alpha=0.25, gamma=2.0, num_classes=4, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            if num_classes == 4:
                self.alpha = torch.tensor([0.25, 1.0, 2.0, 2.0], dtype=torch.float32)
            else:
                self.alpha = torch.ones(num_classes, dtype=torch.float32) * alpha

    def forward(self, inputs, targets):
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        p_t = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1.0 - p_t) ** self.gamma
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        alpha_t = self.alpha[targets]

        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
