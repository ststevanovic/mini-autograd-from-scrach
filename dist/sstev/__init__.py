from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List
import numpy as np
@dataclass
class Tensor:
    data: np.ndarray
    requires_grad: bool = False

    _grad: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _backward: Callable[..., None] = field(default=lambda *args, **kwargs: None,
                                       init=False, repr=False)
    _prev: Tuple["Tensor", ...] = field(default_factory=tuple, init=False, repr=False)
    _op: str = field(default="", init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
        if self.data.dtype.kind != "f":  # enforce floating types for gradients
            self.data = self.data.astype(np.float32)

    @property
    def grad(self) -> Optional[np.ndarray]:
        return self._grad

    def zero_grad(self):
        self._grad = None

    def _ensure_same_dtype(self, other: "Tensor"):
        if self.data.dtype != other.data.dtype:
            raise TypeError(f"dtype mismatch: {self.data.dtype} vs {other.data.dtype}")
    
    # ----- core ops -----

    def __add__(self, other: "Tensor") -> "Tensor":
        self._ensure_same_dtype(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._op, out._prev = "add", (self, other)

        def _backward():
            if self.requires_grad:
                self._accum_grad(self._broadcast_like(out._grad, self.data.shape))
            if other.requires_grad:
                other._accum_grad(self._broadcast_like(out._grad, other.data.shape))
        out._backward = _backward
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        self._ensure_same_dtype(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._op, out._prev = "mul", (self, other)

        def _backward():
            if self.requires_grad:
                self._accum_grad(self._broadcast_like(other.data * out._grad, self.data.shape))
            if other.requires_grad:
                other._accum_grad(self._broadcast_like(self.data * out._grad, other.data.shape))
        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        # matrix multiply: (m,k) @ (k,n) -> (m,n)
        self._ensure_same_dtype(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._op, out._prev = "matmul", (self, other)

        def _backward():
            if self.requires_grad:
                self._accum_grad(out._grad @ other.data.T)
            if other.requires_grad:
                other._accum_grad(self.data.T @ out._grad)
        out._backward = _backward
        return out
    
    def sum(self) -> "Tensor":
        out = Tensor(np.array(self.data.sum(), dtype=np.float32), requires_grad=self.requires_grad)
        out._op, out._prev = "sum", (self,)

        def _backward():
            if self.requires_grad:
                self._accum_grad(np.ones_like(self.data) * out._grad)
        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad)
        out._op, out._prev = "relu", (self,)

        def _backward():
            if self.requires_grad:
                self._accum_grad((self.data > 0).astype(self.data.dtype) * out._grad)
        out._backward = _backward
        return out

    # ----- autograd driver -----

    def backward(self, grad: Optional[np.ndarray] = None):
        """
        Reverse-mode autodiff: seed with dL/d(out).
        If tensor is scalar and no grad provided, seed with 1.
        """
        if not self.requires_grad and grad is None:
            raise RuntimeError("backward on a tensor that does not require grad")

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("non-scalar backward requires grad seed")
            grad = np.ones_like(self.data)

        # topological order (post-order) over the tape
        topo: List[Tensor] = []
        visited = set()

        def build(v: Tensor):
            if id(v) in visited:
                return
            visited.add(id(v))
            for p in v._prev:
                build(p)
            topo.append(v)

        build(self)

        # seed
        self._grad = grad

        # reverse pass
        for v in reversed(topo):
            if v._grad is None:
                continue
            v._backward() 

    # ----- helpers -----

    def _accum_grad(self, g: np.ndarray):
        if self._grad is None:
            self._grad = g
        else:
            self._grad = self._grad + g

    def _broadcast_like(self, grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Sum grad over broadcasted dimensions so it matches target_shape.
        """
        # collapse extra leading dims
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        # sum over axes where target dim == 1 but grad dim > 1
        for i, (gdim, tdim) in enumerate(zip(grad.shape, target_shape)):
            if tdim == 1 and gdim > 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad