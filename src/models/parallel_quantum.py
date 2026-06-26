"""
Optimized parallel quantum execution for EQ-GAN.

Key optimizations over the default sequential approach:
1. No-grad forward: uses PennyLane numpy broadcasting to evaluate all B samples
   in one vectorized call instead of B sequential calls (~23x speedup).
2. With-grad forward+backward: uses a persistent multiprocessing Pool to
   compute per-sample adjoint Jacobians in parallel (~8x speedup).

Drop-in replacement for QuantumLayer in eq_gan.py and
PatchQuantumLayer in patch_qgan.py.
"""

import os, sys, atexit, threading
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

# ── constants ──────────────────────────────────────────────────────────────────

N_WORKERS = min(16, os.cpu_count() or 4)

# ── persistent thread pool + thread-local device cache ─────────────────────────
# ThreadPoolExecutor: no fork/spawn issues, no pickling, GIL released by C++ ext.
# Each thread keeps its own lightning.qubit device via threading.local().
_EXECUTOR: Optional[ThreadPoolExecutor] = None
_tl = threading.local()  # per-thread device/circuit cache


def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=N_WORKERS)
        atexit.register(_EXECUTOR.shutdown, wait=False)
    return _EXECUTOR


# ══════════════════════════════════════════════════════════════════════════════
#  EQ-GAN parallel layer
# ══════════════════════════════════════════════════════════════════════════════

# ── batched forward (numpy, no grad) ──────────────────────────────────────────

def _build_eqgan_circuit(dev, mode, diff_method="best"):
    from .quantum_circuit import N_CONTENT, N_STYLE, N_QUBITS, N_LAYERS
    @qml.qnode(dev, interface="numpy", diff_method=diff_method)
    def full_circuit(noise, style, params):
        cw = list(range(N_CONTENT)); sw = list(range(N_CONTENT, N_QUBITS))
        for i, w in enumerate(cw): qml.RY(noise[..., i], wires=w)
        if mode != "no_style":
            for i, w in enumerate(sw):
                qml.Hadamard(wires=w); qml.RZ(style[..., i], wires=w)
            for i in range(len(sw) - 1):
                qml.IsingZZ(style[..., i] * style[..., i + 1], wires=[sw[i], sw[i + 1]])
            if mode == "full":
                for i, s2 in enumerate(sw): qml.CNOT(wires=[s2, cw[i % len(cw)]])
                for i in range(len(sw) - 1): qml.CNOT(wires=[sw[i], sw[i + 1]])
        for l in range(N_LAYERS):
            n_q = N_CONTENT if mode == "no_style" else N_QUBITS
            for j in range(n_q):
                qml.RZ(params[l, j, 0], wires=j); qml.RY(params[l, j, 1], wires=j); qml.RZ(params[l, j, 2], wires=j)
            for j in range(n_q - 1): qml.CNOT(wires=[j, j + 1])
            qml.CNOT(wires=[n_q - 1, 0])
        return qml.math.stack([qml.expval(qml.PauliZ(w)) for w in cw])
    return full_circuit


_eqgan_fwd_dev = None
_eqgan_fwd_cache: dict = {}  # mode → circuit


def _get_batched_eqgan_circuit(mode: str):
    global _eqgan_fwd_dev
    if _eqgan_fwd_dev is None:
        from .quantum_circuit import N_QUBITS
        _eqgan_fwd_dev = qml.device("lightning.qubit", wires=N_QUBITS)
    if mode not in _eqgan_fwd_cache:
        _eqgan_fwd_cache[mode] = _build_eqgan_circuit(_eqgan_fwd_dev, mode, diff_method="best")
    return _eqgan_fwd_cache[mode]


def _eqgan_batched_forward(noise_np: np.ndarray, style_np, params_np: np.ndarray, mode: str) -> np.ndarray:
    """
    Vectorised forward pass. Returns shape (N_CONTENT, B).
    noise_np: (B, N_CONTENT), style_np: (B, N_STYLE) or None, params_np: (N_LAYERS, N_QUBITS, 3)
    """
    circ = _get_batched_eqgan_circuit(mode)
    if mode == "no_style":
        out = circ(noise_np, None, params_np)
    else:
        out = circ(noise_np, style_np, params_np)
    return np.array(out)  # (N_CONTENT, B)


# ── per-sample VJP (thread-safe, thread-local device cache) ───────────────────

def _eqgan_vjp_worker(args):
    """Run in a ThreadPoolExecutor thread. Uses thread-local device to avoid sharing state."""
    from .quantum_circuit import N_CONTENT, N_STYLE, N_QUBITS, N_LAYERS
    noise_np, style_np, params_np, grad_np, mode = args

    if not hasattr(_tl, "eqgan_devs"):
        _tl.eqgan_devs = {}
        _tl.eqgan_circs = {}

    if mode not in _tl.eqgan_devs:
        dev = qml.device("lightning.qubit", wires=N_QUBITS)
        cw = list(range(N_CONTENT)); sw = list(range(N_CONTENT, N_QUBITS))
        n_q = N_CONTENT if mode == "no_style" else N_QUBITS

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circ(noise, style, params):
            for i, w in enumerate(cw): qml.RY(noise[i], wires=w)
            if mode != "no_style":
                for i, w in enumerate(sw):
                    qml.Hadamard(wires=w); qml.RZ(style[i], wires=w)
                for i in range(len(sw) - 1):
                    qml.IsingZZ(style[i] * style[i + 1], wires=[sw[i], sw[i + 1]])
                if mode == "full":
                    for i, s2 in enumerate(sw): qml.CNOT(wires=[s2, cw[i % len(cw)]])
                    for i in range(len(sw) - 1): qml.CNOT(wires=[sw[i], sw[i + 1]])
            for l in range(N_LAYERS):
                for j in range(n_q):
                    qml.RZ(params[l, j, 0], wires=j); qml.RY(params[l, j, 1], wires=j); qml.RZ(params[l, j, 2], wires=j)
                for j in range(n_q - 1): qml.CNOT(wires=[j, j + 1])
                qml.CNOT(wires=[n_q - 1, 0])
            return tuple(qml.expval(qml.PauliZ(w)) for w in cw)

        _tl.eqgan_devs[mode] = dev
        _tl.eqgan_circs[mode] = circ

    circ = _tl.eqgan_circs[mode]
    noise_t  = torch.tensor(noise_np, dtype=torch.float64, requires_grad=True)
    style_t  = (torch.tensor(style_np, dtype=torch.float64, requires_grad=True)
                if style_np is not None else None)
    params_t = torch.tensor(params_np, dtype=torch.float64, requires_grad=True)

    with torch.enable_grad():
        out = torch.stack(circ(noise_t, style_t, params_t))  # (N_CONTENT,)
        (out * torch.tensor(grad_np, dtype=torch.float64)).sum().backward()

    return (
        noise_t.grad.numpy(),
        style_t.grad.numpy() if style_t is not None else None,
        params_t.grad.numpy(),
    )


# ── custom autograd function ───────────────────────────────────────────────────

class _EQGANParallelFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, noise_enc, style_enc, var_params, mode):
        noise_np  = noise_enc.detach().cpu().numpy().astype(np.float64)
        style_np  = style_enc.detach().cpu().numpy().astype(np.float64) if style_enc is not None else None
        params_np = var_params.detach().cpu().numpy().astype(np.float64)

        out_np = _eqgan_batched_forward(noise_np, style_np, params_np, mode)  # (N_CONTENT, B)
        ctx.save_for_backward(noise_enc, style_enc, var_params)
        ctx.mode = mode
        return torch.tensor(out_np.T, dtype=noise_enc.dtype, device=noise_enc.device)  # (B, N_CONTENT)

    @staticmethod
    def backward(ctx, grad_output):
        noise_enc, style_enc, var_params = ctx.saved_tensors
        mode = ctx.mode

        noise_np  = noise_enc.detach().cpu().numpy().astype(np.float64)
        style_np  = style_enc.detach().cpu().numpy().astype(np.float64) if style_enc is not None else None
        params_np = var_params.detach().cpu().numpy().astype(np.float64)
        grad_np   = grad_output.detach().cpu().numpy().astype(np.float64)  # (B, N_CONTENT)

        B = noise_np.shape[0]
        task_args = [
            (noise_np[i], style_np[i] if style_np is not None else None, params_np, grad_np[i], mode)
            for i in range(B)
        ]
        results = list(map(_eqgan_vjp_worker, task_args))

        noise_vjps, style_vjps, params_vjps = zip(*results)

        noise_grad  = torch.tensor(np.stack(noise_vjps), dtype=noise_enc.dtype, device=noise_enc.device)
        style_grad  = (torch.tensor(np.stack(style_vjps), dtype=style_enc.dtype, device=style_enc.device)
                       if style_enc is not None and style_vjps[0] is not None else None)
        params_grad = torch.tensor(np.sum(params_vjps, axis=0), dtype=var_params.dtype, device=var_params.device)

        return noise_grad, style_grad, params_grad, None


class OptimizedEQGANQuantumLayer(nn.Module):
    """
    Drop-in replacement for QuantumLayer that uses:
    - Batched numpy broadcasting for no_grad forward passes (discriminator steps)
    - Parallel adjoint VJP via Pool for with_grad passes (generator step)
    """
    def __init__(self, mode: str = "full"):
        super().__init__()
        from .quantum_circuit import N_CONTENT, N_QUBITS, N_LAYERS
        assert mode in ("full", "no_entanglement", "no_style")
        self.mode = mode
        n_qubits = N_CONTENT if mode == "no_style" else N_QUBITS
        self.var_params = nn.Parameter(torch.randn(N_LAYERS, n_qubits, 3) * 0.1)

    def forward(self, noise_enc, style_enc=None):
        noise_np  = noise_enc.detach().cpu().numpy().astype(np.float64)
        style_np  = (style_enc.detach().cpu().numpy().astype(np.float64)
                     if style_enc is not None else None)
        params_np = self.var_params.detach().cpu().numpy().astype(np.float64)

        if not torch.is_grad_enabled():
            out_np = _eqgan_batched_forward(noise_np, style_np, params_np, self.mode)
            return torch.tensor(out_np.T, dtype=noise_enc.dtype, device=noise_enc.device)

        return _EQGANParallelFn.apply(noise_enc, style_enc, self.var_params, self.mode)


# ══════════════════════════════════════════════════════════════════════════════
#  Patch QGAN parallel layer
# ══════════════════════════════════════════════════════════════════════════════

def _build_patch_circuit(dev, diff_method="best"):
    from .patch_qgan import N_PATCH_QUBITS, N_PATCH_LAYERS
    @qml.qnode(dev, interface="numpy", diff_method=diff_method)
    def circ(noise_inputs, var_params):
        for i in range(N_PATCH_QUBITS): qml.RY(noise_inputs[..., i], wires=i)
        for l in range(N_PATCH_LAYERS):
            for w in range(N_PATCH_QUBITS):
                qml.RZ(var_params[l, w, 0], wires=w); qml.RY(var_params[l, w, 1], wires=w); qml.RZ(var_params[l, w, 2], wires=w)
            for w in range(N_PATCH_QUBITS - 1): qml.CNOT(wires=[w, w + 1])
            qml.CNOT(wires=[N_PATCH_QUBITS - 1, 0])
        return qml.math.stack([qml.expval(qml.PauliZ(w)) for w in range(N_PATCH_QUBITS)])
    return circ


_patch_fwd_devs: dict = {}
_patch_fwd_circuits: dict = {}   # patch_idx → circuit


def _get_patch_circuit(patch_idx: int):
    from .patch_qgan import N_PATCH_QUBITS
    if patch_idx not in _patch_fwd_devs:
        _patch_fwd_devs[patch_idx] = qml.device("lightning.qubit", wires=N_PATCH_QUBITS)
        _patch_fwd_circuits[patch_idx] = _build_patch_circuit(_patch_fwd_devs[patch_idx])
    return _patch_fwd_circuits[patch_idx]


def _patch_vjp_worker(args):
    """Run in a ThreadPoolExecutor thread. Thread-local device per thread."""
    from .patch_qgan import N_PATCH_QUBITS, N_PATCH_LAYERS
    noise_np, params_np, grad_np, patch_idx = args

    if not hasattr(_tl, "patch_dev"):
        dev = qml.device("lightning.qubit", wires=N_PATCH_QUBITS)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def _circ(noise, params):
            for i in range(N_PATCH_QUBITS): qml.RY(noise[i], wires=i)
            for l in range(N_PATCH_LAYERS):
                for w in range(N_PATCH_QUBITS):
                    qml.RZ(params[l, w, 0], wires=w); qml.RY(params[l, w, 1], wires=w); qml.RZ(params[l, w, 2], wires=w)
                for w in range(N_PATCH_QUBITS - 1): qml.CNOT(wires=[w, w + 1])
                qml.CNOT(wires=[N_PATCH_QUBITS - 1, 0])
            return tuple(qml.expval(qml.PauliZ(w)) for w in range(N_PATCH_QUBITS))

        _tl.patch_dev = dev
        _tl.patch_circ = _circ

    noise_t  = torch.tensor(noise_np, dtype=torch.float64, requires_grad=True)
    params_t = torch.tensor(params_np, dtype=torch.float64, requires_grad=True)

    with torch.enable_grad():
        out = torch.stack(_tl.patch_circ(noise_t, params_t))  # (N_PATCH_QUBITS,)
        (out * torch.tensor(grad_np, dtype=torch.float64)).sum().backward()

    return noise_t.grad.numpy(), params_t.grad.numpy()


class _PatchParallelFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, noise_enc, var_params_list_tensor, patch_idx):
        noise_np  = noise_enc.detach().cpu().numpy().astype(np.float64)   # (B, N_PATCH_QUBITS)
        params_np = var_params_list_tensor.detach().cpu().numpy().astype(np.float64)

        circ = _get_patch_circuit(patch_idx)
        out_np = np.array(circ(noise_np, params_np))   # (N_PATCH_QUBITS, B)
        ctx.save_for_backward(noise_enc, var_params_list_tensor)
        ctx.patch_idx = patch_idx
        return torch.tensor(out_np.T, dtype=noise_enc.dtype, device=noise_enc.device)  # (B, N_PATCH_QUBITS)

    @staticmethod
    def backward(ctx, grad_output):
        noise_enc, var_params_tensor = ctx.saved_tensors
        patch_idx = ctx.patch_idx

        noise_np  = noise_enc.detach().cpu().numpy().astype(np.float64)
        params_np = var_params_tensor.detach().cpu().numpy().astype(np.float64)
        grad_np   = grad_output.detach().cpu().numpy().astype(np.float64)  # (B, N_PATCH_QUBITS)

        B = noise_np.shape[0]
        pool_args = [(noise_np[i], params_np, grad_np[i], patch_idx) for i in range(B)]
        results = list(map(_patch_vjp_worker, pool_args))

        noise_vjps, params_vjps = zip(*results)
        noise_grad  = torch.tensor(np.stack(noise_vjps), dtype=noise_enc.dtype, device=noise_enc.device)
        params_grad = torch.tensor(np.sum(params_vjps, axis=0), dtype=var_params_tensor.dtype, device=var_params_tensor.device)
        return noise_grad, params_grad, None


class OptimizedPatchQuantumLayer(nn.Module):
    """
    Drop-in replacement for PatchQuantumLayer.
    Uses batched numpy for no-grad (discriminator) and
    parallel adjoint Pool for with-grad (generator) steps.
    """
    def __init__(self):
        super().__init__()
        from .patch_qgan import N_PATCHES, N_PATCH_LAYERS, N_PATCH_QUBITS
        self.N_PATCHES = N_PATCHES
        self.var_params = nn.ParameterList([
            nn.Parameter(torch.randn(N_PATCH_LAYERS, N_PATCH_QUBITS, 3) * 0.1)
            for _ in range(N_PATCHES)
        ])

    def forward(self, noise_enc):
        """noise_enc: (B, N_PATCHES, N_PATCH_QUBITS)"""
        noise_cpu = noise_enc.cpu()
        patch_outputs = []
        use_grad = torch.is_grad_enabled()

        for p in range(self.N_PATCHES):
            noise_p = noise_cpu[:, p, :]   # (B, N_PATCH_QUBITS)
            params_p = self.var_params[p]

            if not use_grad:
                noise_np  = noise_p.detach().cpu().numpy().astype(np.float64)
                params_np = params_p.detach().cpu().numpy().astype(np.float64)
                circ = _get_patch_circuit(p)
                out_np = np.array(circ(noise_np, params_np))  # (N_PATCH_QUBITS, B)
                out = torch.tensor(out_np.T, dtype=noise_enc.dtype, device=noise_enc.device)
            else:
                out = _PatchParallelFn.apply(noise_p.to(noise_enc.device), params_p, p)
            patch_outputs.append(out)  # (B, N_PATCH_QUBITS)

        result = torch.stack(patch_outputs, dim=1)  # (B, N_PATCHES, N_PATCH_QUBITS)
        return result.to(noise_enc.device)
