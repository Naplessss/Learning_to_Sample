import torch.nn as nn
import torch


class NormCrossEntropyLoss(object):
    def __init__(self):
        self.loss_op = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, out, data):
        loss = self.loss_op(out, data.y.long())
        loss = loss * data.node_norm
        return loss


class NormBCEWithLogitsLoss(object):
    def __init__(self):
        self.loss_op = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, out, data):
        loss = self.loss_op(out, data.y.type_as(out))
        loss = torch.mul(loss.T, data.node_norm).T
        return loss


class CrossEntropyLoss(object):
    def __init__(self):
        self.loss_op = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, out, data):
        loss = self.loss_op(out, data.y.long()) / data.num_nodes
        return loss

class BCEWithLogitsLoss(object):
    def __init__(self):
        self.loss_op = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, out, data):
        loss = self.loss_op(out, data.y.type_as(out)) / data.num_nodes
        return loss
