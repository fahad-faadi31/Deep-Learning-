import torch
import numpy as np
# 1: SCALAR
print("="*60)
print("SCALAR(0D Tensor)")
print("="*60)
scalar=torch.tensor(100)
print("Tensor:",scalar)
print("Shape:",scalar.shape)
print("Dimensions:",scalar.ndim)
print("Data Type:",scalar.dtype)

# 2: VECTOR (1D)
print("\n"+"="*60)
print("VECTOR(1D Tensor)")
print("="*60)
vector = torch.tensor([10, 20, 30, 40])
print("Tensor:",vector)
print("Shape:",vector.shape)
print("Dimensions:",vector.ndim)
print("Data Type:",vector.dtype)
print("First Element:",vector[0])
print("Third Element:",vector[2])
print("Python Value:",vector[2].item())

# 3: MATRIX (2D Tensor)

print("\n"+"="*60)
print("MATRIX(2D Tensor)")
print("="*60)
matrix=torch.tensor([
    [85,90,78],
    [92,88,95]
])
print(matrix)
print("Shape:",matrix.shape)
print("Dimensions:",matrix.ndim)
print("First Row:")
print(matrix[0])
print("Second Row:")
print(matrix[1])
print("Second Column:")
print(matrix[:, 1])
print("Element [1][2]:",matrix[1][2])

# 4: 3D TENSOR

print("\n"+"="*60)
print("3D TENSOR")
print("="*60)
tensor3d=torch.tensor([
    [
        [1,2],
        [3,4]
    ],
    [
        [5,6],
        [7,8]
    ],
    [
        [9,10],
        [11,12]
    ]
])
print(tensor3d)
print("Shape:",tensor3d.shape)
print("Dimensions:",tensor3d.ndim)
print("First Matrix:")
print(tensor3d[0])
print("Second Matrix Second Row:")
print(tensor3d[1][1])
print("Specific Element:")
print(tensor3d[2][0][1])

# 5: CREATING TENSORS

print("\n"+"="*60)
print("CREATING TENSORS")
print("="*60)
print("Zeros")
print(torch.zeros(2,3))
print("\nOnes")
print(torch.ones(2,3))
print("\nRandom (0 to 1)")
print(torch.rand(2,3))
print("\nRandom Normal")
print(torch.randn(2,3))
print("\nArange")
print(torch.arange(1,11))
print("\nLinspace")
print(torch.linspace(0,10,5))

# 6: TENSOR OPERATIONS

print("\n"+"="*60)
print("TENSOR OPERATIONS")
print("="*60)
a=torch.tensor([1,2,3])
b=torch.tensor([4,5,6])
print("Addition:")
print(a+b)
print("Subtraction:")
print(a-b)
print("Multiplication:")
print(a*b)

print("Division:")
print(b/a)
print("Power:")
print(a**2)

# 7: MATRIX MULTIPLICATION

print("\n"+"="*60)
print("MATRIX MULTIPLICATION")
print("="*60)
A=torch.tensor([
    [1,2],
    [3,4]
])
B=torch.tensor([
    [5,6],
    [7,8]
])
print("Using matmul()")
print(torch.matmul(A,B))
print("\nUsing @ operator")
print(A@B)

# 8: BROADCASTING

print("\n"+"="*60)
print("BROADCASTING")
print("="*60)

matrix=torch.tensor([
    [2,4],
    [6,8]
])
vector=torch.tensor([1,2])
print(matrix+vector)

# 9: RESHAPING

print("\n"+"="*60)
print("RESHAPE")
print("="*60)
x=torch.arange(1,13)
print("Original:")
print(x)
print("Original Shape:",x.shape)
reshaped=x.reshape(3,4)
print("\nReshaped:")
print(reshaped)
print("Reshaped Shape:",reshaped.shape)

# 10: FLATTEN
print("\n"+ "=" * 60)
print("FLATTEN")
print("=" * 60)

matrix = torch.tensor([
    [1, 2],
    [3, 4]
])

print(matrix)

print("Flattened:")
print(matrix.flatten())

# 11: NUMPY ↔ PYTORCH

print("\n"+"="*60)
print("NUMPY <-> PYTORCH")
print("="*60)
array=np.array([1,2,3])
tensor=torch.from_numpy(array)
print("NumPy to Tensor:")
print(tensor)
print("\nTensor to NumPy:")
print(tensor.numpy())
 
# 12: CPU / GPU

print("\n"+"="*60)
print("DEVICE")
print("="*60)
device="cuda" if torch.cuda.is_available() else "cpu"
print("Current Device:",device)
sample=torch.tensor([1,2,3]).to(device)
print(sample)
print(sample.device)