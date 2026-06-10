import torch 

## y = (x+1)^2 

x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()

print(f"y(3) = {y.item():.1f} (expected 16)")
print(f"dy/dx at x =3 {x.grad.item():.1f} (expected 8)")

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(5.0, requires_grad=True)
f = a**2 * b + b**3
f.backward()
print(f"\nf(2,5) = {f.item()} (expected 145)")
print(f"df/da = {a.grad.item()} (expected 20)")
print(f"df/db = {b.grad.item()} (expected 79)")

w = torch.tensor(1.0, requires_grad=True)
for step in range(3):
    loss = (w-2)**2 
    loss.backward()
    print(f"step {step}: w.grad = {w.grad.item()}")

w.grad = None 

with torch.no_grad():
    z = w * 3 
    print(f"\nz inside no_grad -> requires_grad = {z.requires_grad}")

z2 = (w * 3).detach()
print(f"detached z2 -> requires_grad = {z2.requires_grad}")
