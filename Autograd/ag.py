import torch

# 1: requires_grad

print("="*60)
print("REQUIRES_GRAD")
print("="*60)
x=torch.tensor(3.0,requires_grad=True)
print(x)

# 2: BACKWARD & .GRAD

print("\n"+"="*60)
print("BACKWARD & GRAD")
print("="*60)
y=x**2
print("y=",y)
y.backward()
print("Gradient (dy/dx):",x.grad)

# 3: GRADIENT ACCUMULATION

print("\n"+"="*60)
print("GRADIENT ACCUMULATION")
print("="*60)
x=torch.tensor(3.0,requires_grad=True)
y=x**2
y.backward()
print("After first backward():",x.grad)
y.backward()
print("After second backward():",x.grad)

# 4: ZEROING GRADIENTS

print("\n"+"="*60)
print("ZEROING GRADIENTS")
print("="*60)
x=torch.tensor(2.0,requires_grad=True)
y=x**3
y.backward()
print("Gradient:",x.grad)
x.grad.zero_()
print("After zero_():", x.grad)
y.backward()
print("After backward() again:", x.grad)

# 5: COMPUTATION GRAPH

print("\n"+"="*60)
print("COMPUTATION GRAPH")
print("="*60)
x=torch.tensor(2.0,requires_grad=True)
a=x+3
b=a*4
y=b**2
print("a=",a)
print("b=",b)
print("y=",y)
y.backward()
print("dy/dx=",x.grad)

# 6: torch.no_grad()

print("\n"+"="*60)
print("torch.no_grad()")
print("="*60)
x=torch.tensor(3.0,requires_grad=True)
with torch.no_grad():
    y=x**2
print(y)

# 7: detach()

print("\n"+"="*60)
print("detach()")
print("="*60)
x=torch.tensor(4.0,requires_grad=True)
y=x**2
z=y.detach()
print("y.requires_grad =",y.requires_grad)
print("z.requires_grad =",z.requires_grad)