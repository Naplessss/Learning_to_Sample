import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GraphConv


class Filter(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels=1, gcn='sage',drop_out=0.2):
        super(Filter, self).__init__()
        self.in_channels = in_channels
        self.drop_out = drop_out
        self.lin = torch.nn.Linear(hidden_channels*2 , out_channels)
        self.conv1 = {'sage': SAGEConv, 'gat':GATConv, 'graph':GraphConv}.get(gcn)(self.in_channels, hidden_channels)
        self.conv2 = {'sage': SAGEConv, 'gat':GATConv, 'graph':GraphConv}.get(gcn)(hidden_channels, hidden_channels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x, edge_index, edge_weight))
        x1 = F.dropout(x1, p=self.drop_out, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=self.drop_out, training=self.training)
        x = torch.cat([x1,x2], dim=-1)
        x = self.lin(x)
        return torch.sigmoid(x)


class Buffer(nn.Module):

    def __init__(self, num_nodes, num_classes):
        super(Buffer, self).__init__()
        self.register_buffer('prob_each_class', torch.ones(num_nodes, num_classes) / num_classes)
        self.register_buffer('best_valid_loss', torch.ones(num_nodes))
        self.register_buffer('avg_train_loss', torch.zeros(num_nodes))
        self.register_buffer('num_train_loss', torch.zeros(num_nodes))
    
    def update_prob_each_class(self, n_id, new_prob_each_class):
        self.prob_each_class[n_id] = new_prob_each_class.squeeze()


    def update_best_valid_loss(self, n_id, new_valid_loss):
        self.best_valid_loss[n_id] = torch.min(new_valid_loss, self.best_valid_loss[n_id])


    def update_avg_train_loss(self, n_id, train_loss):
        self.avg_train_loss[n_id] = (self.avg_train_loss[n_id] * self.num_train_loss[n_id] + train_loss) / ( self.num_train_loss[n_id] + 1)
        self.num_train_loss[n_id] += 1

    def get_x(self, n_id):
        valid_loss = self.best_valid_loss[n_id].view(-1, 1)
        avg_train_loss = self.avg_train_loss[n_id].view(-1, 1)
        prob_each_class = self.prob_each_class[n_id]
        x = torch.cat([valid_loss, avg_train_loss, prob_each_class], dim=-1)
        return x

