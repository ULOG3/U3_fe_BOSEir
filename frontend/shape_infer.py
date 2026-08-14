"""
Stage 4: Shape & Layout Inference

Walks the (already-legalized) graph in order and computes the shape of every
tensor, using a small shape function per core op — we only need 18 of these,
not one per possible ONNX op, because this runs after legalize().

We do NOT trust whatever shape ONNX's value_info said (see import_stage) —
everything here is recomputed from scratch.
"""

import math


def _conv_out_dim(in_dim, k, pad_before, pad_after, stride, dilation):
    return (in_dim + pad_before + pad_after - dilation * (k - 1) - 1) // stride + 1


def _shape_conv2d(node, graph):
    x, w = node.inputs[0], node.inputs[1]
    N, Cin, H, W = x.shape
    Cout, _, kh, kw = w.shape
    pads = node.attrs.get("pads", [0, 0, 0, 0])          # [top, left, bottom, right]
    strides = node.attrs.get("strides", [1, 1])
    dilations = node.attrs.get("dilations", [1, 1])
    Hout = _conv_out_dim(H, kh, pads[0], pads[2], strides[0], dilations[0])
    Wout = _conv_out_dim(W, kw, pads[1], pads[3], strides[1], dilations[1])
    node.outputs[0].shape = (N, Cout, Hout, Wout)
    node.outputs[0].dtype = x.dtype


def _shape_matmul(node, graph):
    a, b = node.inputs[0].shape, node.inputs[1].shape
    # Simplified: 2D matmul, the common case after Gemm decomposition.
    if len(a) == 2 and len(b) == 2:
        assert a[1] == b[0], f"MatMul '{node.name}': shape mismatch {a} x {b}"
        node.outputs[0].shape = (a[0], b[1])
    else:
        raise NotImplementedError(f"MatMul '{node.name}': only 2D x 2D matmul supported right now (got {a} x {b}).")
    node.outputs[0].dtype = node.inputs[0].dtype


def _broadcast_shape(a, b):
    if a == b:
        return a
    la, lb = len(a), len(b)
    n = max(la, lb)
    a = (1,) * (n - la) + tuple(a)
    b = (1,) * (n - lb) + tuple(b)
    out = []
    for da, db in zip(a, b):
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            raise ValueError(f"Shapes {a} and {b} aren't broadcast-compatible")
    return tuple(out)


def _shape_elementwise_binary(node, graph):
    node.outputs[0].shape = _broadcast_shape(node.inputs[0].shape, node.inputs[1].shape)
    node.outputs[0].dtype = node.inputs[0].dtype


def _shape_same_as_input(node, graph):
    node.outputs[0].shape = node.inputs[0].shape
    node.outputs[0].dtype = node.inputs[0].dtype


def _shape_pool(node, graph):
    x = node.inputs[0]
    N, C, H, W = x.shape
    k = node.attrs["kernel_shape"]
    pads = node.attrs.get("pads", [0, 0, 0, 0])
    strides = node.attrs.get("strides", k)
    Hout = _conv_out_dim(H, k[0], pads[0], pads[2], strides[0], 1)
    Wout = _conv_out_dim(W, k[1], pads[1], pads[3], strides[1], 1)
    node.outputs[0].shape = (N, C, Hout, Wout)
    node.outputs[0].dtype = x.dtype


def _shape_global_avg_pool(node, graph):
    x = node.inputs[0]
    N, C, H, W = x.shape
    node.outputs[0].shape = (N, C, 1, 1)
    node.outputs[0].dtype = x.dtype


def _shape_reshape(node, graph):
    x = node.inputs[0]
    if "flatten_axis" in node.attrs:
        axis = node.attrs["flatten_axis"]
        dim0 = math.prod(x.shape[:axis]) if axis > 0 else 1
        dim1 = math.prod(x.shape[axis:])
        node.outputs[0].shape = (dim0, dim1)
    else:
        # target shape comes from the second input (a Constant)
        target = graph.initializers[node.inputs[1].name].tolist()
        resolved = []
        for i, d in enumerate(target):
            if d == 0:
                resolved.append(x.shape[i])       # 0 means "same as input dim", per ONNX spec
            elif d == -1:
                resolved.append(-1)                # placeholder, fixed below
            else:
                resolved.append(int(d))
        if -1 in resolved:
            known = math.prod([d for d in resolved if d != -1])
            total = math.prod(x.shape)
            resolved[resolved.index(-1)] = total // known
        node.outputs[0].shape = tuple(resolved)
    node.outputs[0].dtype = x.dtype


def _shape_concat(node, graph):
    axis = node.attrs["axis"]
    shapes = [t.shape for t in node.inputs]
    out = list(shapes[0])
    out[axis] = sum(s[axis] for s in shapes)
    node.outputs[0].shape = tuple(out)
    node.outputs[0].dtype = node.inputs[0].dtype


def _shape_transpose(node, graph):
    x = node.inputs[0]
    perm = node.attrs.get("perm", list(reversed(range(len(x.shape)))))
    node.outputs[0].shape = tuple(x.shape[p] for p in perm)
    node.outputs[0].dtype = x.dtype


def _shape_slice(node, graph):
    # Simplified: only handles the common case of constant starts/ends/axes/steps
    # coming in as extra inputs (opset >= 10). Good enough for now — extend if
    # a real model needs attribute-based Slice (opset < 10).
    x = node.inputs[0]
    starts = graph.initializers[node.inputs[1].name].tolist()
    ends = graph.initializers[node.inputs[2].name].tolist()
    axes = graph.initializers[node.inputs[3].name].tolist() if len(node.inputs) > 3 else list(range(len(starts)))
    steps = graph.initializers[node.inputs[4].name].tolist() if len(node.inputs) > 4 else [1] * len(starts)
    shape = list(x.shape)
    for s, e, ax, st in zip(starts, ends, axes, steps):
        dim = shape[ax]
        s = max(0, min(s, dim))
        e = max(0, min(e, dim))
        shape[ax] = max(0, (e - s + (st - 1)) // st)
    node.outputs[0].shape = tuple(shape)
    node.outputs[0].dtype = x.dtype


def _shape_requantize(node, graph):
    x = node.inputs[0]
    node.outputs[0].shape = x.shape
    bitwidth = node.attrs.get("bitwidth")
    signed = node.attrs.get("signed", 1)
    if bitwidth is not None:
        bw = int(bitwidth) if not hasattr(bitwidth, "item") else int(bitwidth.item())
        node.outputs[0].dtype = f"{'int' if signed else 'uint'}{bw}"
    else:
        node.outputs[0].dtype = x.dtype


SHAPE_FNS = {
    "Conv2D": _shape_conv2d,
    "MatMul": _shape_matmul,
    "Add": _shape_elementwise_binary,
    "Mul": _shape_elementwise_binary,
    "Max": _shape_elementwise_binary,
    "Min": _shape_elementwise_binary,
    "ReLU": _shape_same_as_input,
    "Clip": _shape_same_as_input,
    "MaxPool": _shape_pool,
    "AveragePool": _shape_pool,
    "GlobalAveragePool": _shape_global_avg_pool,
    "Reshape": _shape_reshape,
    "Concat": _shape_concat,
    "Transpose": _shape_transpose,
    "Slice": _shape_slice,
    "QuantizeLinear": _shape_same_as_input,
    "DequantizeLinear": _shape_same_as_input,
    "Requantize": _shape_requantize,
    "Identity": _shape_same_as_input,
    "Constant": None,  # already resolved during import — nothing to do
}


def infer_shapes(graph):
    for node in graph.nodes:
        if node.op_type == "Constant":
            continue
        fn = SHAPE_FNS.get(node.op_type)
        if fn is None:
            raise ValueError(f"No shape function registered for core op '{node.op_type}' (node '{node.name}') — "
                              f"this means CORE_OPS and SHAPE_FNS have drifted out of sync.")
        fn(node, graph)
    return graph
