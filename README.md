# Mini Autograd Engine

A minimal automatic differentiation engine built from scratch using only NumPy. Learn how PyTorch and TensorFlow compute gradients by implementing your own autograd system.

## How It Works

This project teaches you the fundamentals of automatic differentiation through hands-on implementation. Instead of just using deep learning frameworks, you'll build the core gradient computation system yourself using only NumPy.

**The Goal:** Understand how autograd works by implementing it yourself in NumPy.

When you complete this project, you'll have built your own working autograd engine that:
- Automatically computes gradients using the chain rule
- Builds computational graphs like PyTorch
- Handles real neural network operations (matrix multiply, ReLU, etc.)

**What You'll Learn:**
- How PyTorch/TensorFlow work under the hood
- The mathematics behind backpropagation
- How computational graphs enable automatic differentiation
- Why reverse-mode autodiff is efficient for neural networks

## The Tensor Class

Your task is to implement a `Tensor` class that tracks operations and computes gradients automatically. Here's the template:

```python
class Tensor:
    def __init__(self, data, requires_grad=False):
        # TODO: implement
        # Store the data (numpy array)
        # Track if gradients are needed
        # Initialize gradient storage
        # Set up graph connections (_prev, _backward, _op)
        raise NotImplementedError
    
    def __add__(self, other):
        # TODO: implement element-wise addition
        # Create output tensor
        # Define backward function for gradient
        # Connect to computation graph
        raise NotImplementedError

    def __mul__(self, other):
        # TODO: implement element-wise multiplication
        raise NotImplementedError

    def __matmul__(self, other):
        # TODO: implement matrix multiplication
        raise NotImplementedError

    def sum(self):
        # TODO: implement sum reduction
        raise NotImplementedError

    def relu(self):
        # TODO: implement ReLU activation
        raise NotImplementedError

    def backward(self, grad=None):
        # TODO: implement backward pass
        # Build topological order of computation graph
        # Seed gradient (start with 1 for scalar outputs)
        # Traverse graph in reverse, calling each _backward()
        raise NotImplementedError
```

### Implementation Steps

**Step 1: Define the Tensor Class**

Your `Tensor` must store:
- `.data` - the actual numpy array
- `.grad` - accumulated gradient
- `.requires_grad` - flag to track if gradients are needed
- `._backward` - function to compute local gradients
- `._prev` - parent nodes (to build the computation graph)

**Step 2: Implement Five Operations**

Add these operations: `__add__`, `__mul__`, `__matmul__`, `sum()`, and `relu()`

Each operation must:
- Create a new output Tensor
- Set `requires_grad=True` if any parent requires gradients
- Define a `_backward()` function to propagate gradients to parents
- Save `_prev` to let `.backward()` traverse the graph

**Step 3: Implement Backward Pass**

Implement `.backward()` to compute all gradients:
- Build a topological ordering of the computation graph
- Seed the final gradient with 1 (for scalar outputs)
- Traverse nodes in reverse order, calling each `_backward()`

## How to Contribute

**Quick Steps:**
1. Fork this repository
2. Create your implementation in a Jupyter notebook: `notebooks/phase-2-2-mini-autograd-engine-<yourname>.ipynb`
3. Test your implementation: `pytest tests/test_autograd.py -v`
4. Open a Pull Request
5. The CI will automatically validate your gradients against PyTorch

**What the CI Tests:**
- Loads your notebook from `notebooks/`
- Extracts your `Tensor` class
- Builds a 2-layer neural network (Linear + ReLU + Linear)
- Compares your gradients against PyTorch's autograd
- If gradients match (within tolerance), your PR is approved!

**Implementation Rules:**
- You can only use Python standard library and NumPy
- PyTorch is provided for testing only, not for your implementation
- Do not use subprocess, os, socket, or other restricted libraries

If you need another library, open an issue to request it.

## Installation

```bash
# Clone the repository
git clone https://github.com/ststevanovic/mini-autograd-from-scrach.git
cd mini-autograd-from-scrach

# Install dependencies
pip install numpy pytest nbformat nbclient ipykernel

# Create your implementation in notebooks/
# Run tests to validate
pytest tests/test_autograd.py -v
```

## Example Usage

Here's how your `Tensor` class should work once implemented:

```python
import numpy as np
from dist.sstev import Tensor  # Reference implementation

# Create tensors
x = Tensor([2.0, 3.0], requires_grad=True)
w = Tensor([4.0, -1.0], requires_grad=True)

# Forward pass
y = x * w              # [8.0, -3.0]
z = y.sum()            # 5.0

# Backward pass - compute gradients automatically
z.backward()

print(x.grad)  # [4.0, -1.0] - gradient w.r.t. x
print(w.grad)  # [2.0, 3.0]  - gradient w.r.t. w
```

## Project Structure

```
mini-autograd-from-scrach/
├── notebooks/              # Your implementations go here
│   └── phase-2-2-mini-autograd-engine-<name>.ipynb
├── dist/sstev/            # Reference implementation
├── tests/                 # Automated gradient tests
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## License

- Code: MIT License
- Course Materials: CC-BY-SA 4.0