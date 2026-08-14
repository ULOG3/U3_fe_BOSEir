"""
Stage 3: Legalize

Converts raw ONNX operations into our chip's 18 supported core operations.
Runs in two phases:
    Phase A — pattern rewrite: fold BatchNorm into the preceding Conv, then
              delete it. Must run BEFORE generic conversion, or BatchNorm
              would get "legalized" into something before the fold pattern
              ever gets a chance to match it.
    Phase B — generic conversion: every remaining node is looked up in a
              registry keyed by (op_type, domain) and either renamed,
              decomposed into a few core ops, or rejected outright with a
              clear error naming the node and why it's unsupported.
"""

import numpy as np
from frontend.bose_ir import Tensor, IRNode

# ---------------------------------------------------------------------------
# Phase A — BatchNorm folding
# ---------------------------------------------------------------------------

def _fold_batchnorm(graph):
    """
    Conv -> BatchNorm, back to back, with nothing else reading Conv's output
    in between: fold BN's scale/shift into Conv's weight/bias, delete BN.

    Math: BN(y) = gamma/sqrt(var+eps) * y + (beta - gamma*mean/sqrt(var+eps))
          y = Conv(x, W, b)
          => new_W = W * (gamma/sqrt(var+eps))     [broadcast over out-channel dim]
          => new_b = (b - mean) * (gamma/sqrt(var+eps)) + beta
    """
    changed = True
    while changed:
        changed = False
        for node in list(graph.nodes):
            if node.op_type != "BatchNormalization":
                continue

            x = node.inputs[0]
            producer = x.producer
            if producer is None or producer.op_type not in ("Conv", "Conv2D"):
                continue  # not directly preceded by a Conv — can't fold
            if len(x.consumers) != 1:
                continue  # Conv's output is used elsewhere too — folding would silently break that other use

            conv = producer
            W_name = conv.inputs[1].name
            W = graph.initializers.get(W_name)
            if W is None:
                continue  # weight isn't a compile-time constant — can't fold

            has_bias = len(conv.inputs) > 2
            b = graph.initializers.get(conv.inputs[2].name) if has_bias else np.zeros(W.shape[0], dtype=W.dtype)

            gamma = graph.initializers[node.inputs[1].name]
            beta = graph.initializers[node.inputs[2].name]
            mean = graph.initializers[node.inputs[3].name]
            var = graph.initializers[node.inputs[4].name]
            eps = node.attrs.get("epsilon", 1e-5)

            factor = gamma / np.sqrt(var + eps)
            new_W = W * factor.reshape(-1, 1, 1, 1)
            new_b = (b - mean) * factor + beta

            _update_constant_value(graph, W_name, new_W)

            if has_bias:
                _update_constant_value(graph, conv.inputs[2].name, new_b)
            else:
                b_tensor = Tensor(conv.name + "_bias_folded", shape=tuple(new_b.shape), dtype=str(new_b.dtype))
                graph.initializers[b_tensor.name] = new_b
                b_const = IRNode("Constant", conv.name + "_bias_folded_const", [], [b_tensor], attrs={"value": new_b})
                graph.nodes.insert(graph.nodes.index(conv), b_const)
                conv.inputs.append(b_tensor)
                b_tensor.consumers.append(conv)

            # Rewire: Conv's real output becomes whatever BN used to output.
            bn_out = node.outputs[0]
            old_conv_out = conv.outputs[0]
            conv.outputs = [bn_out]
            bn_out.producer = conv
            graph.tensors.pop(old_conv_out.name, None)

            # IMPORTANT: detach BEFORE removing, or gamma/beta/mean/var keep a
            # stale reference to this node in their .consumers lists — which
            # would silently hide them from the orphan check in verify().
            _detach(node)
            graph.remove_node(node)
            changed = True
            break  # graph.nodes mutated — restart the scan


def _update_constant_value(graph, tensor_name, new_value):
    graph.initializers[tensor_name] = new_value
    for n in graph.nodes:
        if n.op_type == "Constant" and n.outputs and n.outputs[0].name == tensor_name:
            n.attrs["value"] = new_value
            return


# ---------------------------------------------------------------------------
# Phase B — generic op conversion
# ---------------------------------------------------------------------------

def _rename(new_op_type):
    def convert(node, graph):
        return [IRNode(new_op_type, node.name, node.inputs, node.outputs, dict(node.attrs))]
    return convert


def _convert_flatten(node, graph):
    # Flatten has no shape *input*, just an axis attribute — defer the actual
    # target-shape math to shape_infer (which runs after legalize and will
    # know the resolved input shape by the time it reaches this node).
    axis = node.attrs.get("axis", 1)
    return [IRNode("Reshape", node.name, node.inputs, node.outputs, attrs={"flatten_axis": axis})]


def _convert_gemm(node, graph):
    A, B = node.inputs[0], node.inputs[1]
    C = node.inputs[2] if len(node.inputs) > 2 else None
    attrs = node.attrs
    alpha = attrs.get("alpha", 1.0)
    beta = attrs.get("beta", 1.0)
    transA = attrs.get("transA", 0)
    transB = attrs.get("transB", 0)

    if alpha != 1.0 or (C is not None and beta != 1.0):
        raise ValueError(
            f"Gemm node '{node.name}' uses alpha/beta scaling other than 1.0 — "
            f"not yet supported by legalize() (fail loud rather than compute wrong values)."
        )

    new_nodes = []
    a_in = A
    if transA:
        at = Tensor(node.name + "_A_T")
        new_nodes.append(IRNode("Transpose", node.name + "_transposeA", [A], [at], attrs={"perm": [1, 0]}))
        a_in = at

    b_in = B
    if transB:
        bt = Tensor(node.name + "_B_T")
        new_nodes.append(IRNode("Transpose", node.name + "_transposeB", [B], [bt], attrs={"perm": [1, 0]}))
        b_in = bt

    if C is not None:
        mm_out = Tensor(node.name + "_mm")
        new_nodes.append(IRNode("MatMul", node.name + "_matmul", [a_in, b_in], [mm_out], attrs={}))
        new_nodes.append(IRNode("Add", node.name + "_bias", [mm_out, C], node.outputs, attrs={}))
    else:
        new_nodes.append(IRNode("MatMul", node.name + "_matmul", [a_in, b_in], node.outputs, attrs={}))

    return new_nodes


def _convert_qonnx_quant(node, graph):
    x = node.inputs[0]
    scale_t = node.inputs[1] if len(node.inputs) > 1 else None
    zeropt_t = node.inputs[2] if len(node.inputs) > 2 else None
    bitwidth_t = node.inputs[3] if len(node.inputs) > 3 else None

    def const_val(t):
        return graph.initializers.get(t.name) if t is not None else None

    scale, zeropt, bitwidth = const_val(scale_t), const_val(zeropt_t), const_val(bitwidth_t)
    if scale is None or zeropt is None:
        raise ValueError(
            f"Quant node '{node.name}': scale/zero-point aren't compile-time constants — "
            f"dynamic quantization parameters aren't supported yet."
        )
    attrs = {
        "scale": scale, "zero_point": zeropt, "bitwidth": bitwidth,
        "signed": node.attrs.get("signed", 1),
        "narrow": node.attrs.get("narrow", 0),
        "rounding_mode": node.attrs.get("rounding_mode", "ROUND"),
    }
    return [IRNode("Requantize", node.name, [x], node.outputs, attrs=attrs)]


def _convert_qonnx_bipolar_quant(node, graph):
    x = node.inputs[0]
    scale_t = node.inputs[1] if len(node.inputs) > 1 else None
    scale = graph.initializers.get(scale_t.name) if scale_t is not None else None
    attrs = {"scale": scale, "zero_point": 0, "bitwidth": 1, "signed": 1, "bipolar": True}
    return [IRNode("Requantize", node.name, [x], node.outputs, attrs=attrs)]


def _convert_clip_v1(node, graph):
    """
    Opset < 11: min/max are ATTRIBUTES on the node (optional, default -inf/+inf).
    We normalize to the same {"min": float, "max": float} attrs regardless of
    which ONNX version we imported from — everything downstream only ever
    needs to know one representation.
    """
    min_val = node.attrs.get("min", float("-inf"))
    max_val = node.attrs.get("max", float("inf"))
    return [IRNode("Clip", node.name, node.inputs[:1], node.outputs, attrs={"min": min_val, "max": max_val})]


def _convert_clip_v11(node, graph):
    """
    Opset >= 11: min/max are OPTIONAL TENSOR INPUTS (2nd, 3rd inputs) instead
    of attributes. They must be compile-time constants for us — resolve them
    from graph.initializers into the same normalized {"min", "max"} attrs.
    """
    x = node.inputs[0]
    min_val, max_val = float("-inf"), float("inf")
    if len(node.inputs) > 1 and node.inputs[1].name in graph.initializers:
        min_val = float(graph.initializers[node.inputs[1].name])
    elif len(node.inputs) > 1:
        raise ValueError(f"Clip node '{node.name}': min bound isn't a compile-time constant — not supported.")
    if len(node.inputs) > 2 and node.inputs[2].name in graph.initializers:
        max_val = float(graph.initializers[node.inputs[2].name])
    elif len(node.inputs) > 2:
        raise ValueError(f"Clip node '{node.name}': max bound isn't a compile-time constant — not supported.")
    return [IRNode("Clip", node.name, [x], node.outputs, attrs={"min": min_val, "max": max_val})]


# Ops whose legalization genuinely differs by opset version, à la TVM's
# _impl_vX pattern: {op_type: {min_version_this_impl_applies_from: converter}}.
# At lookup time we pick the highest registered version <= the model's opset.
VERSIONED_CONVERT_MAP = {
    "Clip": {1: _convert_clip_v1, 11: _convert_clip_v11},
}


# Standard ONNX ops (domain == "") -> converter function
CONVERT_MAP = {
    "Relu": _rename("ReLU"),
    "Conv": _rename("Conv2D"),
    "MatMul": _rename("MatMul"),
    "Add": _rename("Add"),
    "Mul": _rename("Mul"),
    "Max": _rename("Max"),
    "Min": _rename("Min"),
    "MaxPool": _rename("MaxPool"),
    "AveragePool": _rename("AveragePool"),
    "GlobalAveragePool": _rename("GlobalAveragePool"),
    "Reshape": _rename("Reshape"),
    "Concat": _rename("Concat"),
    "Slice": _rename("Slice"),
    "Transpose": _rename("Transpose"),
    "Identity": _rename("Identity"),
    "QuantizeLinear": _rename("QuantizeLinear"),
    "DequantizeLinear": _rename("DequantizeLinear"),
    "Flatten": _convert_flatten,
    "Gemm": _convert_gemm,
    # NOTE: "BatchNormalization" is intentionally absent — it must be
    # removed by _fold_batchnorm before this table is ever consulted. If one
    # somehow survives (no foldable preceding Conv), it correctly falls
    # through to rejection below — we have no runtime BN op in our ISA.
}

# QONNX custom-domain ops -> converter function
QONNX_CONVERT_MAP = {
    "Quant": _convert_qonnx_quant,
    "BipolarQuant": _convert_qonnx_bipolar_quant,
}

# Known-unsupported ops get a specific reason instead of a generic message
EXCLUDED_REASONS = {
    "Sigmoid": "needs a dedicated nonlinearity unit not present in our 18-op ISA",
    "Softmax": "needed only for attention/transformer models — out of scope for the current CNN-classifier target",
    "Tanh": "needs a dedicated nonlinearity unit not present in our 18-op ISA",
    "LSTM": "recurrent ops need stateful control flow our static-dataflow ISA doesn't support",
    "GRU": "recurrent ops need stateful control flow our static-dataflow ISA doesn't support",
    "If": "dynamic control flow isn't supported — the backend assumes a fully static graph",
    "Loop": "dynamic control flow isn't supported — the backend assumes a fully static graph",
    "NonMaxSuppression": "detection post-processing needs dynamic control flow — out of scope for classifier models",
}


def _lookup_converter(node, opset):
    if node.domain and node.op_type in QONNX_CONVERT_MAP:
        return QONNX_CONVERT_MAP[node.op_type]
    if node.domain:
        return None

    if node.op_type in VERSIONED_CONVERT_MAP:
        versions = VERSIONED_CONVERT_MAP[node.op_type]
        applicable = [v for v in versions if opset is None or v <= opset]
        if not applicable:
            return None  # model's opset is older than any impl we have for this op
        return versions[max(applicable)]

    return CONVERT_MAP.get(node.op_type)


def _detach(node):
    for t in node.inputs:
        if node in t.consumers:
            t.consumers.remove(node)
    for t in node.outputs:
        if t.producer is node:
            t.producer = None


def legalize(graph):
    _fold_batchnorm(graph)

    new_nodes = []
    for node in graph.nodes:
        if node.op_type == "Constant":
            new_nodes.append(node)
            continue

        converter = _lookup_converter(node, graph.opset)
        if converter is None:
            reason = EXCLUDED_REASONS.get(node.op_type, "no legalization rule exists for it")
            raise ValueError(
                f"Unsupported op '{node.op_type}' at node '{node.name}': {reason}. "
                f"This model can't be compiled for our chip as-is."
            )

        _detach(node)
        new_nodes.extend(converter(node, graph))

    graph.nodes = new_nodes
    for n in graph.nodes:
        for t in n.outputs:
            graph.tensors[t.name] = t

    _prune_dead_constants(graph)
    return graph


def _prune_dead_constants(graph):
    """
    Some converters (e.g. Clip v11) resolve a constant tensor INTO an
    attribute rather than passing it through as a runtime input — e.g.
    min_t/max_t become node.attrs["min"]/["max"] instead of staying as
    graph inputs. Their Constant nodes are now genuinely unused. This is
    normal, expected dead code — prune it here rather than let verify()
    fail on something that isn't actually a bug.
    """
    graph_output_names = {t.name for t in graph.outputs}
    kept = []
    for n in graph.nodes:
        if n.op_type == "Constant" and len(n.outputs[0].consumers) == 0 and n.outputs[0].name not in graph_output_names:
            graph.tensors.pop(n.outputs[0].name, None)
            continue
        kept.append(n)
    graph.nodes = kept
