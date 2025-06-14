import rdkit 
from torch_geometric.datasets import MoleculeNet
import torch 
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling,global_mean_pool
from torch_geometric.nn import global_mean_pool as gap,global_max_pool as gmp

data = MoleculeNet(root="Data-1",name="ESOL")
embedding_size = 64

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN,self).__init__()
        torch.manual_seed(42)

        self.initial_conv = GCNConv(data.num_features,embedding_size)
        self.conv1 = GCNConv(embedding_size,embedding_size)
        self.conv2 = GCNConv(embedding_size,embedding_size)
        self.conv3 = GCNConv(embedding_size,embedding_size)
        self.out = Linear(embedding_size*2,1)

    def forward(self,x,edge_index,batch_index):
        hidden = self.initial_conv(x,edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden,edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden,edge_index)
        hidden = F.tanh(hidden)

        hidden = torch.cat([gmp(hidden,batch_index),
                            gap(hidden,batch_index)],dim=1)
        out = self.out(hidden)

        return out, hidden

model =GCN()
print(model)

from torch_geometric.data import DataLoader

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0007)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data_size = len(data)
batch_size = 64

loader = DataLoader(data[:int(data_size*0.8)],
                    batch_size =batch_size,shuffle=True)

test_loader = DataLoader(data[int(data_size*0.8):],
                         batch_size=batch_size,shuffle=True)


def train(data):
    for batch in loader:
        batch.to(device)
        optimizer.zero_grad()
        pred,embedding = model(batch.x.float(),batch.edge_index, batch.batch)
        loss = criterion(pred,batch.y)
        loss.backward()
        optimizer.step()

    return loss,embedding


print(
        "starting traning"
        )

losses = []

for epch in range(2000):
    loss, h = train(data)
    losses.append(loss)
    if epch%100 == 0:
        print(f"Epoch {epch} | Train loss {loss}")










