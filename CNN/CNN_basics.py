



import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 


class CNN(nn.Module):
    def __init__(self,in_cahnnel=1, num_classes=10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        # N-out = (N-in + 2*Padding - Kernel_size)/Stride + 1
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv3 = nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size=3,stride = 1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(32*25*25,num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x

model = CNN()
x = torch.randn(64,1,100,100)

print(model(x).shape)




class Simple1DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Simple1DCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        
        self.relu = nn.ReLU()
        
        self.fc = nn.Linear(32 * 64, num_classes)
        
    def forward(self, x):

        x = self.conv1(x)  
        x = self.relu(x)
        
        x = self.conv2(x)  
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)  
        
        x = self.fc(x)             
        
        return x



inp = torch.randn(64,1,64)
model = Simple1DCNN()
print(model(inp).shape)



state = {"state_dict": model.state_dict()}
torch.save(state,"saving.pth.tar")

model.load_state_dict(torch.load("saving.pth.tar")["state_dict"])
