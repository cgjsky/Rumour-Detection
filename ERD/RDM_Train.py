from model import *
import torch
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
epochs=5
for epoch in range(epochs):
    for i,(x,date,y) in enumerate(loader):
        y=y.reshape(-1)
        optim.zero_grad()
        out=rdm_model(x)
        out=out.argmax(axis=1)
        out=torch.tensor(out,dtype=torch.float32,requires_grad=True)
        loss=criteration(out,y)
            
        loss.backward()
        optim.step()
        if i % 50 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == y).sum().item() / len(y)
            print(i, loss.item(), accuracy)