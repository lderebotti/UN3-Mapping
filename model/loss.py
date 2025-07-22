import torch
import torch.nn as nn
import math


def sdf_diff_loss(pred, label, weight, scale=1.0, l2_loss=True):
    count = pred.shape[0]
    diff = pred - label
    diff_m = diff / scale  # so it's still in m unit
    if l2_loss:
        loss = (weight * (diff_m**2)).sum() / count  # l2 loss
    else:
        loss = (weight * torch.abs(diff_m)).sum() / count  # l1 loss
    return loss

def sdf_l1_loss(pred, label):
    loss = torch.abs(pred - label)
    return loss.mean()

def sdf_l2_loss(pred, label):
    loss = (pred - label) ** 2
    return loss.mean()

# used by our approach
def heter_sdf_loss(pred, label, sigma, log_sigma):
    loss = torch.abs(pred - label) / sigma + log_sigma
    return loss.mean()

# only learn uncertainty, not backprop to sdf field.
# more stable for general scenes.
def heter_detach_sdf_loss(pred, label, sigma, log_sigma):
    pred = pred.detach()
    loss = torch.abs(pred - label) / sigma + log_sigma
    return loss.mean()

def color_diff_loss(pred, label, weight, weighted=False, l2_loss=False):
    diff = pred - label
    if not weighted:
        weight = 1.0
    else:
        weight = weight.unsqueeze(1)
    if l2_loss:
        loss = (weight * (diff**2)).mean()
    else:
        loss = (weight * torch.abs(diff)).mean()
    return loss

# BCE loss adapted from SHINE-Mapping
def sdf_bce_loss(pred, label, beta, weight, weighted=False, bce_reduction="mean"):
    if weighted:
        loss_bce = nn.BCEWithLogitsLoss(reduction=bce_reduction, weight=weight)
    else:
        loss_bce = nn.BCEWithLogitsLoss(reduction=bce_reduction)
    label_op = torch.sigmoid(label / beta)  # occupancy prob
    loss = loss_bce(pred / beta, label_op)
    return loss

def heter_bce_loss_v1(pred, label, beta, sigma, log_sigma):
    loss_bce = nn.BCEWithLogitsLoss(reduction="mean")
    label_op = torch.sigmoid(label / beta)
    loss = loss_bce(pred / beta, label_op)/sigma + log_sigma
    return loss

def heter_bce_loss_v2(pred, label, sigma, log_sigma):
    loss_bce = nn.BCEWithLogitsLoss(reduction="mean")
    label_op = torch.sigmoid(label / sigma)
    loss = loss_bce(pred / sigma, label_op)
    print(sigma)
    return loss

def heter_bce_loss_v3(pred, label, beta, sigma, log_sigma):
    loss_bce = nn.BCEWithLogitsLoss(reduction="mean")
    label_op = torch.sigmoid(label / beta)  # occupancy prob
    # directly add noise to predicted sdf 
    noise = sigma * torch.randn_like(pred)  # Gaussian noise
    pred = pred + noise  # Add noise to logit
    loss = loss_bce(pred / beta, label_op)
    return loss