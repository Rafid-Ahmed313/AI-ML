import torch
from torch.optim import Adam


X = torch.tensor([5.0,2.0],requires_grad=True)

y = 100

z = 50

optimizer = Adam([X], lr = 1)

for epoch in range(300):
    optimizer.zero_grad()
    loss = (X[0]-y)**2 + (X[1]-z)**2
    loss.backward()
    optimizer.step()
    print(f"Value: {X}, Loss: {loss}")

x = torch.tensor(torch.randn(2,2),requires_grad=True)
m = torch.tensor(([1,0],[0,1]))

optimizer = Adam([x], lr =0.1)
for epoch in range(300):
    optimizer.zero_grad()
    loss = (x[0][0] -m[0][0])**2 + (x[0][1] - m[0][1])**2 + (x[1][0]-m[1][0])**2 + (x[1][1]-m[1][1])**2
    loss.backward()
    optimizer.step()
    print(f"Value: {x}, loss: {loss}")

print(x)



