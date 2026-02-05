# 🧮 Mini Autograd Engine

A minimal yet powerful reverse-mode automatic differentiation (autograd) engine built from scratch using only NumPy. This project is part of the **ML Systems Bootcamp** and serves as an educational tool to understand how deep learning frameworks like PyTorch and TensorFlow compute gradients.

## 📖 Table of Contents
- [What is Autograd?](#what-is-autograd)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [API Documentation](#api-documentation)
- [Implementation Guide](#implementation-guide)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🤔 What is Autograd?

Automatic differentiation (autograd) is the core technology behind modern deep learning. Instead of manually computing derivatives, autograd automatically calculates gradients of complex functions by applying the chain rule. This engine implements **reverse-mode** autodiff, which is efficient for functions with many inputs and few outputs (like neural network loss functions).

### Why Build This?

Understanding autograd helps you:
- 🧠 Deeply understand how PyTorch/TensorFlow work under the hood
- 🐛 Debug gradient-related issues in your models
- 🚀 Appreciate the engineering behind deep learning frameworks
- 🎓 Master fundamental ML systems concepts

## ✨ Features

- ✅ **Core Operations**: Add, Multiply, Matrix Multiply, ReLU, Sum
- ✅ **Automatic Gradient Computation**: Reverse-mode autodiff with `.backward()`
- ✅ **Broadcasting Support**: Handles NumPy-style broadcasting correctly
- ✅ **Computational Graph**: Builds and traverses computation graphs automatically
- ✅ **Pure NumPy**: No deep learning frameworks required for implementation
- ✅ **PyTorch-Compatible API**: Similar interface to PyTorch tensors

## 🚀 Installation

### Using the Prebuilt Package

```bash
# Clone the repository
git clone https://github.com/ststevanovic/mini-autograd-from-scrach.git
cd mini-autograd-from-scrach

# Install dependencies
pip install -r requirements.txt

# Use the prebuilt Tensor class
from dist.sstev import Tensor
```

### For Development/Learning

```bash
# Install dependencies
pip install numpy pytest nbformat nbclient ipykernel

# Create your own implementation in notebooks/
# See docs/how-to-PR.md for contribution guidelines
```

## ⚡ Quick Start

Here's a simple example showing how to compute gradients:

```python
import numpy as np
from dist.sstev import Tensor

# Create tensors
x = Tensor([2.0, 3.0], requires_grad=True)
w = Tensor([4.0, -1.0], requires_grad=True)

# Forward pass: compute y = x * w, then sum
y = x * w              # Element-wise: [8.0, -3.0]
z = y.sum()            # Scalar: 5.0

# Backward pass: compute gradients
z.backward()

print(f"x.grad: {x.grad}")  # [4.0, -1.0] - gradient w.r.t. x
print(f"w.grad: {w.grad}")  # [2.0, 3.0]  - gradient w.r.t. w
```

### Simple Neural Network Example

```python
# Tiny linear layer with ReLU
x = Tensor([[1.0, 2.0, 3.0]], requires_grad=False)     # Input (1, 3)
W = Tensor(np.random.randn(3, 2), requires_grad=True)  # Weights (3, 2)

# Forward: y = ReLU(x @ W)
h = x @ W
y = h.relu()

# Compute loss (sum for simplicity)
loss = y.sum()

# Backward: compute dL/dW
loss.backward()

print(f"Weight gradient shape: {W.grad.shape}")  # (3, 2)
print(f"Weight gradients:\n{W.grad}")
```

## 🔧 How It Works

### 1. **Computational Graph**

Every operation builds a directed acyclic graph (DAG):

```python
x = Tensor([2.0])
w = Tensor([3.0], requires_grad=True)
y = x * w      # Creates node: y depends on (x, w)
z = y + y      # Creates node: z depends on (y, y)
```

```
Graph:  x ──┐
            ├──> y ──> z
        w ──┘      ┌──┘
                   └────┘
```

### 2. **Forward Pass**

Operations store:
- `data`: The computed result
- `_prev`: Parent tensors
- `_backward`: Function to compute local gradients
- `_op`: Operation name (for debugging)

```python
def __mul__(self, other):
    out = Tensor(self.data * other.data, 
                 requires_grad=self.requires_grad or other.requires_grad)
    out._op, out._prev = "mul", (self, other)
    
    def _backward():
        if self.requires_grad:
            self._accum_grad(other.data * out._grad)  # d/dself = other
        if other.requires_grad:
            other._accum_grad(self.data * out._grad)  # d/dother = self
    
    out._backward = _backward
    return out
```

### 3. **Backward Pass**

`.backward()` performs reverse-mode autodiff:

1. **Topological Sort**: Order nodes from inputs to outputs
2. **Seed Gradient**: Start with `∂L/∂L = 1` at the loss
3. **Reverse Traversal**: Apply chain rule, calling each `_backward()`
4. **Gradient Accumulation**: Sum gradients from multiple paths

```python
def backward(self, grad=None):
    # 1. Build topological order
    topo = []
    visited = set()
    
    def build(v):
        if id(v) in visited:
            return
        visited.add(id(v))
        for parent in v._prev:
            build(parent)
        topo.append(v)
    
    build(self)
    
    # 2. Seed gradient
    self._grad = grad if grad is not None else np.ones_like(self.data)
    
    # 3. Reverse pass
    for node in reversed(topo):
        if node._grad is not None:
            node._backward()
```

### 4. **The Chain Rule in Action**

For `loss = ((x @ W).relu()).sum()`:

```
Backward:
loss ──> sum ──> relu ──> matmul ──> x, W
  1        1      mask     W.T@    
                           @.T(x)
```

Each operation knows its local derivative and applies the chain rule automatically!

## 📚 API Documentation

### Tensor Class

```python
Tensor(data, requires_grad=False)
```

**Parameters:**
- `data`: Array-like or numpy array
- `requires_grad`: Whether to track gradients (default: `False`)

**Attributes:**
- `.data`: The underlying NumPy array
- `.grad`: Accumulated gradient (or `None`)
- `.requires_grad`: Gradient tracking flag

**Methods:**

#### `backward(grad=None)`
Computes gradients via reverse-mode autodiff.

```python
loss = model(x)
loss.backward()  # Computes gradients for all requires_grad=True tensors
```

#### `zero_grad()`
Resets accumulated gradients to `None`.

```python
optimizer_step()
tensor.zero_grad()  # Clear before next backward pass
```

### Operations

#### Arithmetic Operations

```python
z = x + y    # Element-wise addition
z = x * y    # Element-wise multiplication
z = x @ y    # Matrix multiplication
```

#### Activation Functions

```python
y = x.relu()  # ReLU: max(0, x)
```

#### Reduction Operations

```python
s = x.sum()   # Sum all elements to scalar
```

### Broadcasting

The engine correctly handles NumPy broadcasting:

```python
x = Tensor([[1.0, 2.0, 3.0]])  # Shape: (1, 3)
y = Tensor([[1.0],             # Shape: (3, 1)
            [2.0], 
            [3.0]])

z = x + y  # Broadcasts to (3, 3), gradients correctly back-propagate
```

## 💻 Implementation Guide

### Step 1: Define Tensor Class
Your `Tensor` must store:
- `.data`: the actual numpy array
- `.grad`: accumulated gradient
- `.requires_grad`: flag
- `_backward`: local backward function
- `_prev`: parent nodes (to build the graph)

### Step 2: Add five ops: Add, Mul, Matmul, Sum and ReLU
Each op must:
- Return a new Tensor with requires_grad=True if any parent requires it.
- Define _backward to propagate gradients to parents.
- Save _prev to let .backward() traverse the graph.

### Step 3: Add one op: Backward
Implement `.backward()`:
- Do a reverse topological traversal of the graph.
- Seed final grad with 1 if scalar.
- Call each node's `_backward()` in reverse order.

### Template

Here's a quick template to get started:

```python
class Tensor:
    def __init__(self, data, requires_grad=False):
        # TODO: implement
        raise NotImplementedError
    
    def __add__(self, other):
        # TODO: implement add
        raise NotImplementedError

    def __mul__(self, other):
        # TODO: implement
        raise NotImplementedError

    def __matmul__(self, other):
        # TODO: implement
        raise NotImplementedError

    def sum(self, other):
        # TODO: implement
        raise NotImplementedError

    def relu(self, other):
        # TODO: implement
        raise NotImplementedError

    def backward(self, grad=None):
        # TODO: implement backward
        raise NotImplementedError
```

## 📂 Project Structure

```
mini-autograd-from-scrach/
├── notebooks/              # Student implementations
│   └── phase-2-2-mini-autograd-engine-<name>.ipynb
├── dist/                   # Prebuilt packages
│   └── sstev/             # Reference implementation
│       └── __init__.py    # Complete Tensor class
├── tests/                  # Automated tests
│   └── test_autograd.py   # PyTorch comparison tests
├── docs/                   # Documentation
│   └── how-to-PR.md       # Contribution guide
├── .github/workflows/      # CI/CD
│   └── ci.yml             # Automated testing
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🚀 Goal (For Bootcamp Students)

Students must implement their own `Tensor` class with the following operations:
- ✅ `__add__` - Element-wise addition
- ✅ `__mul__` - Element-wise multiplication  
- ✅ `__matmul__` - Matrix multiplication
- ✅ `relu()` - ReLU activation
- ✅ `sum()` - Sum reduction
- ✅ `backward()` - Reverse-mode autodiff

Your implementation should pass PyTorch gradient comparison tests!

## 🧪 Testing

The CI automatically tests your implementation:

1. Loads your notebook from `notebooks/`
2. Extracts your `Tensor` class
3. Builds a 2-layer MLP (Linear + ReLU + Linear)
4. Compares gradients against PyTorch's autograd
5. ✅ If gradients match (within tolerance) → PR approved!

Run tests locally:
```bash
pytest tests/test_autograd.py -v
```

### Environment Rules
- ✅ **Allowed**: Python stdlib + NumPy
- ✅ **Provided**: PyTorch (for testing only, not for your implementation)
- ❌ **Forbidden**: Using PyTorch/TensorFlow autograd, subprocess, os, socket, etc.

If you need another library, open an issue and request it.

## 🤝 Contributing

We welcome contributions! See [docs/how-to-PR.md](docs/how-to-PR.md) for detailed guidelines.

**Quick steps:**
1. Fork this repository
2. Create your implementation in `notebooks/phase-2-2-mini-autograd-engine-<yourname>.ipynb`
3. Ensure tests pass: `pytest tests/`
4. Open a Pull Request
5. Wait for CI to validate your gradients

## 📜 License

- **Code**: MIT License
- **Course Materials** (notebooks, docs): CC-BY-SA 4.0

---

**Built with ❤️ for the ML Systems Bootcamp**

*Learn by building the tools you use every day!*