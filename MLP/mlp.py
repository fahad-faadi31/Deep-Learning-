import torch
import torch.nn as nn
X=torch.tensor([
    [0.0,0.0],
    [0.0,1.0],
    [1.0,0.0],
    [1.0,1.0]])
y=torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]])

model=nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid()
)
print(model)
criterion=nn.BCELoss()
optimizer=torch.optim.SGD(
    model.parameters(),
    lr=0.1)

epochs = 5000

for epoch in range(epochs):
    optimizer.zero_grad()
    output=model(X)
    loss=criterion(output,y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch{epoch+1},Loss:{loss.item():.4f}")

with torch.no_grad():
    predictions=model(X)
print(predictions)