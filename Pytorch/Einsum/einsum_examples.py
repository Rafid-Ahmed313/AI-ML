import torch 

x = torch.randn(2,3)

torch.einsum("ij->ji",x)

torch.einsum("ij->",x)

torch.einsum("ij->j",x)

torch.einsum("ij->i",x)

v = torch.randn(1,3)

torch.einsum("ij, kj->ik",x,v)

torch.einsum("ij,kj->ik",x,x)

torch.einsum("i,i->",x[0],x[0])
torch.einsum("ij,ij->",x,x)

torch.einsum("ij,ij->ij",x,x)

a = torch.randn(3)
b = torch.randn(5)

torch.einsum("i,j-ij",a,b)







