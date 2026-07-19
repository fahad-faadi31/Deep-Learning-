import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Custom Dataset

class XORDataset(Dataset):
    def __init__(self):
        self.X=torch.tensor([
            [0.0,0.0],
            [0.0,1.0],
            [1.0,0.0],
            [1.0,1.0]])
        self.y=torch.tensor([
            [0.0],
            [1.0],
            [1.0],
            [0.0]])
    def __len__(self):
        return len(self.X)
    def __getitem__(self,index):
        return self.X[index],self.y[index]

# Dataset & DataLoader

dataset=XORDataset()
dataloader=DataLoader(dataset,batch_size=2,shuffle=True)

# Build MLP Model

model=nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid())

# Loss Function

criterion=nn.BCELoss()

# Optimizer

optimizer=optim.SGD(model.parameters(),lr=0.1)

# Training Loop

epochs=1000
for epoch in range(epochs):
    for batch_X,batch_y in dataloader:
        # Forward Pass
        output=model(batch_X)
        # Calculate Loss
        loss=criterion(output,batch_y)
        # Clear Old Gradients
        optimizer.zero_grad()
        # Backward Pass
        loss.backward()
        # Update Weights
        optimizer.step()
    if (epoch+1)%100==0:
        print(f"Epoch{epoch+1},Loss:{loss.item():.4f}")