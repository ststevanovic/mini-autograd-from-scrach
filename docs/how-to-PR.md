# How to contribute
1. Fork this repo.  
2. Add your Tensor implementaion to `notebooks/` 
    _(see README for additional detail)_
    
    ### Step 1: Define Tensor Class
    Your `Tensor` must store:
    - `.data`: the actual numpy array
    - `.grad`: accumulated gradient
    - `.requires_grad`: flag
    - `_backward`: local backward function
    - `_prev`: parent nodes (to build the graph)
    <br/>

    ### Step 2: Add five ops: Add, Mul, Matmul, Sum and ReLU
    Each op must:
    - Return a new Tensor with requires_grad=True if any parent requires it.
    - Define _backward to propagate gradients to parents.
    - Save _prev to let .backward() traverse the graph.
    <br/>

    ### Step 3: Add one op: Backward
    Implement `.backward()`:
    - Do a reverse topological traversal of the graph.
    - Seed final grad with 1 if scalar.
    - Call each node’s `_backward()` in reverse order.
    <br/>

    Here's a quick template:
    
    ```copy-paste
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


3. Open a PR.

Big Thanks!!!
