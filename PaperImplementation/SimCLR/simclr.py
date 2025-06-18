import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import AdamW
from math import cos, pi


class SimCLRAug(torch.nn.Module):
    def __init__(self):
        super().__init__()
        color_jitter = transforms.ColorJitter(.4,.4,.4,.1)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2,1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=.8),
            transforms.RandomGrayscale(p=.2),
            transforms.GaussianBlur(3, sigma=(.1,2.)),
            transforms.ToTensor()
        ])
    def forward(self, x):
        return self.train_transform(x), self.train_transform(x)

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.base = datasets.CIFAR10(root, download=True)
        self.aug = SimCLRAug()
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        img, _ = self.base[i]
        x1, x2 = self.aug(img)
        return x1, x2


class SimCLR(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()          # 512-d output
        self.proj = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim, bias=False)
        )
    def forward(self, x):
        h = self.encoder(x)
        z = F.normalize(self.proj(h), dim=1)
        return z



def nt_xent(z: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    B = z.size(0)
    assert B%2==0
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.t()) / tau
    sim.fill_diagonal_(float('-inf'))
    labels = torch.arange(B, device=z.device) ^ 1
    loss = F.cross_entropy(sim, labels)
    return loss



device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimCLR().to(device)
opt = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
ds = ContrastiveDataset("./data")
loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, drop_last=True)

epochs = 100
warm = 10
for epoch in range(epochs):
    for x1,x2 in loader:
        x = torch.cat([x1,x2]).to(device)
        loss = nt_xent(model(x))
        loss.backward()
        opt.step(); opt.zero_grad()
    
    lr = 3e-4 * .5 * (1 + cos(pi * epoch / epochs))
    for pg in opt.param_groups: pg["lr"] = lr
    print(f"{epoch:03d} | loss {loss.item():.3f}")

torch.save(model.encoder.state_dict(), "simclr_cifar.pt")

