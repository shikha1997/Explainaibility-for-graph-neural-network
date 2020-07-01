import torch
import torch.nn as nn
from torchviz import make_dot
from utils import *
from rdkit import Chem
import torch.nn.functional as F
from layers import GraphConvolution, GraphAggregation


#from torch.distributions import Categor
class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout):
        super(Generator, self).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        layers = []
        for c0, c1 in zip([z_dim]+conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout)

    def forward(self, x):
        output = self.layers(x)
        edges_logits = self.edges_layer(output)\
                       .view(-1,self.edges,self.vertexes,self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0,1,3,2))/2
        edges_logits = self.dropoout(edges_logits.permute(0,2,3,1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1,self.vertexes,self.nodes))

        return edges_logits, nodes_logits

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim, m_dim, b_dim, dropout):
        super(Discriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, dropout)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1], aux_dim, b_dim, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0,c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activatation=None):
        adj = adj[:,:,:,1:].permute(0,3,1,2)
        annotations = torch.cat((hidden, node), -1) if hidden is not None else node
        h = self.gcn_layer(annotations, adj)
        annotations = torch.cat((h, hidden, node) if hidden is not None\
                                 else (h, node), -1)
        h = self.agg_layer(annotations, torch.tanh)
        h = self.linear_layer(h)

        # Need to implemente batch discriminator #
        ##########################################

        output = self.output_layer(h)
        output = activatation(output) if activatation is not None else output

        return output, h

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    out = torch.zeros(list(labels.size()) + [dim])
    out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
    return out

def postprocess(inputs, method, temperature=1.):

    def listify(x):
        return x if type(x) == list or type(x) == tuple else [x]

    def delistify(x):
        return x if len(x) > 1 else x[0]

    if method == 'soft_gumbel':
        softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
    elif method == 'hard_gumbel':
        softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
    else:
        softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

    return [delistify(e) for e in (softmax)]

def reward(metric,mols):
    rr = 1.
    for m in ('logp,sas,qed,unique' if metric == 'all' else metric).split(','):

        if m == 'np':
            rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
        elif m == 'logp':
            rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
        elif m == 'sas':
            rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
        elif m == 'qed':
            rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
        elif m == 'novelty':
            rr *= MolecularMetrics.novel_scores(mols, data)
        elif m == 'dc':
            rr *= MolecularMetrics.drugcandidate_scores(mols, data)
        elif m == 'unique':
            rr *= MolecularMetrics.unique_scores(mols)
        elif m == 'diversity':
            rr *= MolecularMetrics.diversity_scores(mols, data)
        elif m == 'validity':
            rr *= MolecularMetrics.valid_scores(mols)
        else:
            raise RuntimeError('{} is not defined as a metric'.format(m))

    return rr.reshape(-1, 1)

def matrices2mol(mols,node_labels, edge_labels, strict=False):
    mol = Chem.RWMol()
    atom_labels = sorted(set([atom.GetAtomicNum() for mol in mols for atom in mol.GetAtoms()] + [0]))
    print(atom_labels)
    #atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
    atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
    print(atom_decoder_m)
    bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType()
                                                                for mol in mols
                                                                for bond in mol.GetBonds())))
    print(bond_labels)

    #bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
    bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
    print(bond_decoder_m)
    print("node_labels",node_labels)
    print("edge_labels",edge_labels)
    # node_labels=[node_labels]
    # edge_labels = [edge_labels]
    # node_labels =node_labels.unsqueeze(dim=0)
    # edge_labels = edge_labels.unsqueeze(dim=0)
    for node_label in node_labels:
        print(node_label)
        mol.AddAtom(Chem.Atom(atom_decoder_m[node_label]))

    for start, end in zip(*np.nonzero(edge_labels)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[edge_labels[start, end]])

    if strict:
        try:
            Chem.SanitizeMol(mol)
        except:
            mol = None

    return mol




gen_model = Generator([128, 256, 512], 8, 9, 5,5, 0.3)
z = torch.rand(1,8).float()
a = torch.rand(1,9,9).long()           # Adjacency.
x = torch.rand(1,9).long()# Nodes.
a_tensor = label2onehot(a,5)
x_tensor = label2onehot(x, 5)
print("z",z.shape,"a",a.shape,"x",x.shape,"a_tensor",a_tensor.shape,"x_tensor",x_tensor.shape)
edge, node = gen_model(z)
print(edge.shape,node.shape)
(edges_hat,nodes_hat)= postprocess((edge,node),method= 'hard_gumbel')
edges_hat=edges_hat.unsqueeze(dim=0)
nodes_hat = nodes_hat.unsqueeze(dim=0)
dis_model = Discriminator( [[128, 64], 128, [128, 64]],5,5, 0.3)
vis_model= Discriminator([[128, 64], 128, [128, 64]],5,5,0.3)
logits_fake, features_fake = dis_model(edges_hat, None, nodes_hat)
g_loss_fake = - torch.mean(logits_fake)

mols = [Chem.MolFromSmiles('Cc1ccccc1')]
rewardR = torch.from_numpy(reward(metric='all',mols=mols))

(edges_hard, nodes_hard) = postprocess((edge, node), 'hard_gumbel')
edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
print("ed",edges_hard.shape,"nd",nodes_hard.shape)
#---Error
print([(n_,e_) for e_, n_ in zip(edges_hard, nodes_hard)])
mols = [matrices2mol(mols,n_, e_, strict=True) for e_, n_ in zip(edges_hard, nodes_hard)]
rewardF = torch.from_numpy(reward(mols))
print("rewardF")
# Value loss
value_logit_real, _ = vis_model(a_tensor, None, x_tensor, torch.sigmoid)
value_logit_fake, _ = vis_model(edges_hat, None, nodes_hat, torch.sigmoid)
g_loss_value = torch.mean((value_logit_real - rewardR) ** 2)# + (value_logit_fake - rewardF) ** 2)
# rl_loss= -value_logit_fake
# f_loss = (torch.mean(features_real, 0) - torch.mean(features_fake, 0)) ** 2
print(g_loss_fake)

g_optimizer = torch.optim.Adam(list(gen_model.parameters())+list(vis_model.parameters()),
                                            0.0001, [0.5, 0.999])
# Backward and optimize.
g_loss = g_loss_fake + g_loss_value

print("g_loss", g_loss)

g_optimizer.zero_grad()
g_loss.backward()
g_optimizer.step()

# Logging.
loss['G/loss_fake'] = g_loss_fake.item()
loss['G/loss_value'] = g_loss_value.item()





























