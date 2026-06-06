import torch 

# a tensor is just a n-dimensional array, we are making a 2D shape (2,2), the 1 makes it a float32, the default type for deep learning - integers wont have gradients, you almost always want floats 

# Shape is the elements along each axis, device is where the bytes livev (GPU/CPU), dtype is the kind of number dimensions algin form the right; each pair must e equal or one of them must be 1

# Multiplications are used for masks, gates and attention weights, @ is matrix multiplication, A linear layer ix x @ W.T+ b, Every neural network is a chain of @ calls with non-linearities sprinkled between.

def section(title):
    print(f"{title}")

section("create")
x = torch.tensor([[1., 2.],
                    [3., 4.]])


print(x)
print("shape:", x.shape, "| dtype:", x.dtype, "| device:", x.device)
 
section("broadcasting")
y = torch.tensor([10., 20.])
print(x + y)
section("reshape")
print(x.view(4))
print(x.T)

section("multiply")
print(x*x)
print(x@x)

# Collapsing a dimension
section("reductions")
print("sum:", x.sum().item())
print("mean over rows (dim=0)", x.mean(dim=0))
print("mean over cols (dim=1)", x.mean(dim=1))

section("device")
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print("device:", device)
x_gpu = x.to(device)
print("now on:", x_gpu.device)


