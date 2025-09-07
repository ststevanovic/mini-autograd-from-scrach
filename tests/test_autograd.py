import nbformat
from nbclient import NotebookClient
import pathlib
import inspect
import zipfile
import hashlib
import random
import shutil
import numpy as np
import torch

def save_tensor_as_whl(Tensor, nb_path: str):
    """Extract source from Tensor class and package into a whl-like archive."""
    code = inspect.getsource(Tensor)

    nb_file = pathlib.Path(nb_path).stem  # e.g. phase-2-2-mini-autograd-engine-alice
    student_name = nb_file.split("-")[-1]

    # make 7-digit hash from student name + randomness
    seed = f"{student_name}-{random.random()}"
    short_hash = hashlib.sha1(seed.encode()).hexdigest()[:7]

    out_dir = pathlib.Path("dist")
    out_dir.mkdir(parents=True, exist_ok=True)

    pkgdir = out_dir / f"{student_name}"
    if pkgdir.exists():
        shutil.rmtree(pkgdir)
    pkgdir.mkdir()  # clean slate
    (pkgdir / "__init__.py").write_text(code)

    whl_name = f"{short_hash}-autograd-engine-{student_name}.whl"
    whl_path = out_dir / whl_name

    with zipfile.ZipFile(whl_path, "w") as z:
        for file in pkgdir.rglob("*"):
            z.write(file, arcname=str(file.relative_to(out_dir)))

    print(f"Created {whl_path}")

def load_tensor_from_notebook(nb_path:str):
    """
    Execute a notebook and return its Tensor class.
    Assumes notebook defines a class named `Tensor`.
    """
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=700, kernel_name="python3")
    client.execute()

    ns = {}

    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, ns)
    if "Tensor" not in ns:
        raise RuntimeError(f"No Tensor class found in {nb_path}")
    
    Tensor = ns["Tensor"]

    # Archive into wheel-like zip
    save_tensor_as_whl(Tensor, nb_path)
    
    return Tensor


def test_linear_relu_mlp():
    notebooks_dir = pathlib.Path("notebooks")
    notebooks = list(notebooks_dir.glob("*.ipynb"))
    assert notebooks, "No student notebooks found"

    for nb in notebooks:
        Tensor = load_tensor_from_notebook(str(nb))

        np.random.seed(42)

        x  = np.random.randn(4, 3).astype(np.float32)  # (n, m)
        W1 = np.random.randn(3, 5).astype(np.float32)  # (m, p)
        W2 = np.random.randn(5, 2).astype(np.float32)  # (p, k)
        t  = np.random.randn(4, 2).astype(np.float32)  # (n, k)

        # student engine
        xT  = Tensor(x)
        W1T = Tensor(W1, requires_grad=True)
        W2T = Tensor(W2, requires_grad=True)

        h = (xT @ W1T).relu()
        y = h @ W2T
        loss = ((y - Tensor(t)) * (y - Tensor(t))).sum()
        loss.backward()

        # pytorch reference
        xP  = torch.tensor(x, dtype=torch.float32)
        W1P = torch.tensor(W1, dtype=torch.float32, requires_grad=True)
        W2P = torch.tensor(W2, dtype=torch.float32, requires_grad=True)
        tP  = torch.tensor(t, dtype=torch.float32)

        hP = torch.relu(xP @ W1P)
        yP = hP @ W2P
        lossP = ((yP - tP) ** 2).sum()
        lossP.backward()

        assert np.allclose(
            W1T.grad, W1P.grad.numpy(), atol=1e-5
        ), f"{nb.name}: W1 grads mismatch"
        assert np.allclose(
            W2T.grad, W2P.grad.numpy(), atol=1e-5
        ), f"{nb.name}: W2 grads mismatch"
