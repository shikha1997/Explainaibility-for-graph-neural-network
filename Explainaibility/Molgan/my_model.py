import torch
#from torch_geometric.data import Data
from torch import nn,optim
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SGConv
#import torch_scatter.scatter_cuda
class GCN_network(torch.nn.Module):
    def __init__(self,in_ch, hid1, hid2, hid3, l1_hid,l2_hid,final, drop, K):
    # def __init__(self,in_ch, hid1,hid3, l1_hid,l2_hid, drop, K):
        super(GCN_network,self).__init__()
        self.drop=drop
        self.conv1=SGConv(in_ch,hid1,K)
        self.conv2=SGConv(hid1,hid2,K)
        self.conv3=GATConv(hid2,hid3)

        self.l1=nn.Linear(10*hid3,l1_hid)
        self.l2=nn.Linear(l1_hid,l2_hid)
        self.l3=nn.Linear(l2_hid,final)   #p450-cyp2c19-Potency','p450-cyp2c9-Potency','p450-cyp1a2-Potency','p450-cyp2d6-Potency','p450-cyp3a4-Potency'
        self.l=nn.LeakyReLU(0.1)
    def forward(self,x,edge_index):
        x=x
        edge_index=edge_index
        x=self.conv1(x,edge_index)
        self.grad_value = x.clone()
        x=self.l(x)
        x = F.dropout(x, self.drop, training=self.training)
        x=self.l(self.conv2(x,edge_index))
        x=self.l(self.conv3(x,edge_index))
        x=x.view(int(len(x)/10),-1)
        x=self.l(self.l1(x))
        x=F.dropout(x,self.drop,training=self.training)
        x=self.l(self.l2(x))
        x=F.dropout(x,self.drop, training=self.training)
        x=self.l3(x)
        #self.grad_value = x.clone()
        return x

    def grad_get(self):
        return self.grad_value

if __name__ == '__main__':
    model = GCN_network(46, 128, 256, 128, 512, 128, 5, 0.5, 2)
    print(model())
