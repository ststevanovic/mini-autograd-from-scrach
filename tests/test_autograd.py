import nbformat
from nbclient import NotebookClient
import pathlib
import zipfile
import hashlib
import random
import shutil
import numpy as np
import torch


def save_tensor_as_whl(nb_path: str, code:str):
    """
    Extract Tensor class source and its import lines from a notebook.
    Freeze those into a whl-like archive for tracking.
    """
    nb = nbformat.read(nb_path, as_version=4)
    tensor_code = []
    import_lines = set()
    found_tensor = False

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source

        # Collect all import statements (freeze what student used)
        for line in src.splitlines():
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                import_lines.add(line)

        # Grab Tensor class definition(s)
        if "class Tensor" in src:
            found_tensor = True
            tensor_code.append(src)

    if not found_tensor:
        raise RuntimeError("Tensor class not found in notebook for packaging.")

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


def run_safeguard(nb):
    """
    Execute only safe cells from a notebook.
    Check imports against the actual CI environment.
    Returns (Tensor class, combined source).
    """
    ns = {}
    banned = ("subprocess", "os.", "shutil", "sys.", "socket")

    tensor_src = []
    import_lines = []

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source.strip()

        # skip magics / shell
        if src.startswith("!") or src.startswith("%"):
            continue

        # check banned dangerous stuff
        if any(b in src for b in banned):
            raise RuntimeError(f"Banned module/function detected in code: {src}")

        # validate imports against current environment
        for line in src.splitlines():
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                mod = line.split()[1].split(".")[0]
                try:
                    __import__(mod)
                except ImportError:
                    raise RuntimeError(
                        f"Import '{mod}' not available in course environment. "
                        f"Please stick to standard library + numpy (see README)."
                    )
                import_lines.append(line)

        # Capture Tensor class
        if "class Tensor" in src:
            tensor_src.append(src)

        exec(src, ns)

    if "Tensor" not in ns:
        raise RuntimeError("No Tensor class found in notebook")

    return ns["Tensor"], "\n".join(import_lines + tensor_src)



def load_tensor_from_notebook(nb_path: str):
    """
    Parse a notebook safely, return its Tensor class,
    and archive its implementation into a whl-like zip.
    """
    nb = nbformat.read(nb_path, as_version=4)

    Tensor, code = run_safeguard(nb)

    # Archive into wheel-like zip with captured code
    save_tensor_as_whl(nb_path, code)

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

        # (y - t) implemented as y + (-1 * t)
        neg_t = Tensor(np.array(-1.0, dtype=np.float32)) * Tensor(t)
        diff = y + neg_t
        loss = (diff * diff).sum()  # requires student's .sum()
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
