# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 10:20:25 2021

@author: bcz
"""

import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TDRDGAT(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,heads):
        super(TDRDGAT, self).__init__()
        self.gat1 = GATConv(in_feats, hid_feats,heads,concat=True)
        self.gat2 = GATConv(hid_feats*heads, out_feats,heads,concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = scatter_mean(x, data.batch, dim=0)  #(batch*out_feats)   
        return x

class BURDGAT(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,heads):
        super(BURDGAT, self).__init__()
        self.gat1 = GATConv(in_feats, hid_feats,heads,concat=True)
        self.gat2 = GATConv(hid_feats*heads, out_feats,heads,concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = scatter_mean(x, data.batch, dim=0)  #(batch*out_feats)   
        return x

class Net(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,heads):
        super(Net, self).__init__()
        self.TDRDGAT = TDRDGAT(in_feats, hid_feats, out_feats,heads)
        self.BURDGAT = BURDGAT(in_feats, hid_feats, out_feats,heads)
        self.fc=th.nn.Linear(out_feats+hid_feats,4)

    def forward(self, data):
        TD_x = self.TDRDGAT(data)
        BU_x = self.BURDGAT(data)
        x = th.cat((BU_x,TD_x), 1)
        x=self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
