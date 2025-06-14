from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

data = Planetoid(root="data/Planetoid", name = "Cora", transform = NormalizeFeatures())


import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv 

class GCN(torch.nn.Module):
    def __init__(self,hidden_channels):
        super(GCN,self).__init__()
        torch.manual_seed(42)

        self.conv1 = GCNConv(data.num_features,hidden_channels)
        self.conv2 = GCNConv(hidden_channels,hidden_channels)
        self.out = Linear(hidden_channels,data.num_classes)

    def forward(self,x,edge_index):
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = F.dropout(x,p=0.5,training=self.training)

        x = self.conv2(x,edge_index)
        x = x.relu()
        x = F.dropout(x,p=0.5,training= self.training)

        x = F.softmax(self.out(x),dim=1)
        return x


model = GCN(hidden_channels = 16)

print(model)





model = GCN(hidden_channels=16)

device = torch.device("cudo:0" if torch.cuda.is_available()else "cpu")
model = model.to(device)
data = data.to(device)

learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate,weight_decay = decay)

criterion = torch.nn.CrossEntropyLoss()
print("Data inference")
print(data.x.shape)
print(data.edge_index)
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x,data.edge_index)
    loss = criterion(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test():
    model.eval()
    out = model(data.x,data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


losses = []
for epoch in range(0,1001):
    loss = train()
    losses.append(loss)
    if epoch%100 == 0 :
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
