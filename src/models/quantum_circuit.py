"""
EQ-GAN quantum circuit: 8 qubits total (6 content + 2 style) + entanglement core.

Design rationale:
- 8 qubits is NISQ-feasible (tested on real hardware in related work)
- 6 content qubits → 64-dim quantum state space for generation
- 2 style qubits → IQP feature map for conditional class control
- Entanglement core: CNOT ladder style→content (the key contribution)
- Backend: lightning.qubit (CPU); 8 qubits → fast per-sample evaluation
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np


N_CONTENT = 6
N_STYLE   = 2
N_QUBITS  = N_CONTENT + N_STYLE   # 8 total
N_LAYERS  = 3


_dev = qml.device("lightning.qubit", wires=N_QUBITS)


def _angle_encoding(inputs, wires):
    for i, w in enumerate(wires):
        qml.RY(inputs[i], wires=w)


def _iqp_feature_map(inputs, wires):
    """IQP-style feature map: H → Rz → ZZ interactions."""
    for i, w in enumerate(wires):
        qml.Hadamard(wires=w)
        qml.RZ(inputs[i], wires=w)
    for i in range(len(wires) - 1):
        qml.IsingZZ(inputs[i] * inputs[i + 1], wires=[wires[i], wires[i + 1]])


def _hardware_efficient_layer(params, wires, layer_idx):
    """Rz-Ry-Rz + CNOT ring."""
    n = len(wires)
    for i, w in enumerate(wires):
        qml.RZ(params[layer_idx, i, 0], wires=w)
        qml.RY(params[layer_idx, i, 1], wires=w)
        qml.RZ(params[layer_idx, i, 2], wires=w)
    for i in range(n - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    qml.CNOT(wires=[wires[-1], wires[0]])


def _entanglement_core(content_wires, style_wires):
    """
    Entanglement core: CNOT ladder from style register → content register.
    Creates cross-register quantum correlations that carry conditional
    style information into the content generation subspace.
    """
    for i, sw in enumerate(style_wires):
        cw = content_wires[i % len(content_wires)]
        qml.CNOT(wires=[sw, cw])
    # Intra-style entanglement
    for i in range(len(style_wires) - 1):
        qml.CNOT(wires=[style_wires[i], style_wires[i + 1]])


@qml.qnode(_dev, interface="torch", diff_method="adjoint")
def eq_gan_circuit(noise_inputs, style_inputs, var_params):
    """
    Full EQ-GAN circuit (mode='full').
    noise_inputs: (N_CONTENT,), style_inputs: (N_STYLE,),
    var_params: (N_LAYERS, N_QUBITS, 3)
    Returns: [N_CONTENT Pauli-Z expectation values]
    """
    content_wires = list(range(N_CONTENT))
    style_wires   = list(range(N_CONTENT, N_QUBITS))

    _angle_encoding(noise_inputs, content_wires)
    _iqp_feature_map(style_inputs, style_wires)
    _entanglement_core(content_wires, style_wires)

    for l in range(N_LAYERS):
        _hardware_efficient_layer(var_params, list(range(N_QUBITS)), l)

    return [qml.expval(qml.PauliZ(w)) for w in content_wires]


@qml.qnode(_dev, interface="torch", diff_method="adjoint")
def ablation_no_entanglement_circuit(noise_inputs, style_inputs, var_params):
    """Ablation: identical circuit WITHOUT the entanglement core."""
    content_wires = list(range(N_CONTENT))
    style_wires   = list(range(N_CONTENT, N_QUBITS))

    _angle_encoding(noise_inputs, content_wires)
    _iqp_feature_map(style_inputs, style_wires)
    # No _entanglement_core call

    for l in range(N_LAYERS):
        _hardware_efficient_layer(var_params, list(range(N_QUBITS)), l)

    return [qml.expval(qml.PauliZ(w)) for w in content_wires]


_dev_content = qml.device("lightning.qubit", wires=N_CONTENT)


@qml.qnode(_dev_content, interface="torch", diff_method="adjoint")
def ablation_no_style_circuit(noise_inputs, var_params):
    """Ablation: content-only circuit, no style register (unconditional)."""
    content_wires = list(range(N_CONTENT))
    _angle_encoding(noise_inputs, content_wires)
    for l in range(N_LAYERS):
        _hardware_efficient_layer(var_params[:, :N_CONTENT, :], content_wires, l)
    return [qml.expval(qml.PauliZ(w)) for w in content_wires]


def compute_entanglement_entropy(var_params: torch.Tensor,
                                 n_samples: int = 50) -> float:
    """
    Estimate von Neumann entanglement entropy between content and style
    registers by sampling random input states and averaging.
    """
    dev_sv = qml.device("lightning.qubit", wires=N_QUBITS)
    entropies = []

    for _ in range(n_samples):
        noise = torch.rand(N_CONTENT) * 2 * np.pi
        style = torch.rand(N_STYLE)  * 2 * np.pi

        @qml.qnode(dev_sv)
        def sv_circuit():
            _angle_encoding(noise, list(range(N_CONTENT)))
            _iqp_feature_map(style, list(range(N_CONTENT, N_QUBITS)))
            _entanglement_core(list(range(N_CONTENT)), list(range(N_CONTENT, N_QUBITS)))
            for l in range(N_LAYERS):
                _hardware_efficient_layer(
                    var_params.detach(), list(range(N_QUBITS)), l
                )
            return qml.state()

        sv = np.array(sv_circuit())
        # Bipartition: content (first N_CONTENT qubits) vs style (rest)
        rho = sv.reshape(2 ** N_CONTENT, 2 ** N_STYLE)
        sv_vals = np.linalg.svd(rho, compute_uv=False)
        sv_vals = sv_vals[sv_vals > 1e-12]
        probs = sv_vals ** 2
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        entropies.append(entropy)

    return float(np.mean(entropies))
