import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
import torchvision 
import torchvision.transforms as transforms 
import math 
import matplotlib.pyplot as plt 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


transform = transforms.Compose([
    transforms.ToTensor()
    ])


train_dataset = torchvision.datasets.MNIST(root='./data', train=True,transform=transform,download=True)

train_loader = DataLoader(train_dataset,batch_size=128,shuffle=True)

T = 200
beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(beta_start,beta_end,T).to(device)
alphas = 1.0 - betas 
alpha_bars = torch.cumprod(alphas,dim=0)

def q_sample(x_start,t,noise=None):
    
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alpha_bar_t = torch.sqrt(alpha_bars[t])[:,None,None,None]
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1-alpha_bars[t])[:,None,None,None]

    return sqrt_alpha_bar_t*x_start + sqrt_one_minus_alpha_bar_t * noise


def p_losses(model,x_start,t):
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start,t,noise=noise)
    predicted_noise = model(x_noisy,t)
    return nn.functional.mse_loss(predicted_noise,noise)



def sinusodial_embedding(timesteps,dim=32):
    half_dim = dim//2 
    freq = torch.exp(
            -math.log(10000) * torch.arange(0,half_dim,dtype=torch.float32,device=timesteps.device)/half_dim
            )

    args = timesteps[:,None].float() * freq[None]

    embedding = torch.cat([torch.sin(args),torch.cos(args)],dim=-1)
    return embedding 




class SimpleUNet(nn.Module):
    def __init__(self, time_dim=32):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.ReLU()
        )
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.conv_middle = nn.Conv2d(64, 64, 3, padding=1)
        
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1)
        
        self.act = nn.ReLU()
        
        self.time_layer1 = nn.Linear(128, 32)
        self.time_layer2 = nn.Linear(128, 64)

    def forward(self, x, t):
        
        t_emb = sinusoidal_embedding(t)       # (B, time_dim)
        t_emb = self.time_mlp(t_emb)          # (B, 128)
        
        x = self.act(self.conv1(x))
        x = x + self.time_layer1(t_emb)[:, :, None, None]
        x = self.act(self.conv2(x))
        
        x = x + self.time_layer2(t_emb)[:, :, None, None]
        x = self.act(self.conv_middle(x))
        
        x = self.act(self.deconv2(x))
        x = self.deconv1(x)  # no activation to predict noise, could also add Tanh
        
        return x

model = SimpleUNet().to(device)



optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 3

model.train()
for epoch in range(epochs):
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        
        t = torch.randint(0, T, (images.shape[0],), device=device).long()
        
        loss = p_losses(model, images, t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx} Loss: {loss.item():.4f}")




@torch.no_grad()
def p_sample(model, x, t):
  
    betas_t = betas[t]
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bars[t])
    sqrt_inv_alpha_t = 1.0 / torch.sqrt(alphas[t])
    
    model_mean = sqrt_inv_alpha_t * ( x - betas_t / sqrt_one_minus_alpha_bar_t * model(x, t) )
    
    if t > 0:
        z = torch.randn_like(x)
        sigma_t = torch.sqrt(betas_t)
        return model_mean + sigma_t * z
    else:
        return model_mean

@torch.no_grad()
def p_sample_loop(model, shape):
 
    x = torch.randn(shape, device=device)
    for i in reversed(range(T)):
        x = p_sample(model, x, torch.tensor([i], device=device, dtype=torch.long))
    return x

model.eval()
sampled_images = p_sample_loop(model, (16, 1, 28, 28))  # generate 16 images
sampled_images = sampled_images.clamp(-1, 1)  # if needed


import math

plt.figure(figsize=(6,6))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(sampled_images[i].cpu().squeeze(), cmap="gray")
    plt.axis('off')
plt.tight_layout()
plt.show()

