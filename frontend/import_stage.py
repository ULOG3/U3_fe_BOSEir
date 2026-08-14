"""
Stage 1 + 2: Validate and Import

This is the very first thing that touches the .onnx file. Import is a pure,
dumb, 1:1 transcription — every ONNX node becomes one IRNode with the exact
same op_type it had in ONNX. No semantic conversion happens here on purpose;
that's the legalize stage's job. Keeping this stage "dumb" means bugs are
easy to localize.
"""

import onnx
from onnx import numpy_helper

from frontend.bose_ir import BoseIR, Tensor, IRNode

# ONNX's numeric dtype codes -> simple string names we use internally.
_DTYPE_MAP = {
    1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64",
    9: "bool", 10: "float16", 11: "float64",
}


def validate(onnx_path: str) -> onnx.ModelProto:
    """
    Load the .onnx file and run ONNX's own built-in checker on it.
    Catches a broken/malformed file BEFORE we try to do anything smart
    with it — if this fails, the file itself is the problem, not our code.
    """
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    return model


def _get_opset_version(model: onnx.ModelProto) -> int:
    """
    A .onnx file can technically import several opsets (one per domain).
    We want the version of the MAIN ONNX opset specifically (domain "" or
    "ai.onnx") — that's the one that governs standard op semantics like
    Clip's min/max representation.
    """
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    return None  # unusual — no main-domain opset declared


def _onnx_attrs_to_dict(node: onnx.NodeProto) -> dict:
    """Convert ONNX's attribute list into a plain python dict."""
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def import_model(model: onnx.ModelProto) -> BoseIR:
    """
    Walk model.graph.node / initializer / input / output and build our own
    BoseIR/IRNode/Tensor objects — a direct 1:1 transcription of the ONNX
    graph. No op-conversion logic here; that's legalize.legalize().
    """
    onnx_graph = model.graph
    graph = BoseIR(name=onnx_graph.name or "model")
    graph.opset = _get_opset_version(model)

    # 1. Constants (weights, biases, scale factors, ...)
    for init in onnx_graph.initializer:
        arr = numpy_helper.to_array(init)
        graph.initializers[init.name] = arr
        t = graph.get_tensor(init.name)
        t.shape = tuple(arr.shape)
        t.dtype = str(arr.dtype)
        # Constants get their own IRNode so they're visible in the graph,
        # not just floating data — this matches the "Constant" core op.
        const_node = IRNode("Constant", name=f"{init.name}_const", inputs=[], outputs=[t],
                             attrs={"value": arr})
        graph.add_node(const_node)

    # 2. Graph inputs (skip any that are actually initializers, ONNX allows both)
    initializer_names = {i.name for i in onnx_graph.initializer}
    for inp in onnx_graph.input:
        if inp.name in initializer_names:
            continue
        t = graph.get_tensor(inp.name)
        t.shape, t.dtype = _shape_dtype_from_value_info(inp)
        graph.inputs.append(t)

    # 3. Nodes, in the order ONNX gives them (ONNX requires topological order)
    for onnx_node in onnx_graph.node:
        in_tensors = [graph.get_tensor(n) for n in onnx_node.input if n != ""]
        out_tensors = [graph.get_tensor(n) for n in onnx_node.output]
        attrs = _onnx_attrs_to_dict(onnx_node)
        node = IRNode(
            op_type=onnx_node.op_type,
            name=onnx_node.name or f"{onnx_node.op_type}_{len(graph.nodes)}",
            inputs=in_tensors,
            outputs=out_tensors,
            attrs=attrs,
            domain=onnx_node.domain,  # "" for standard ONNX, non-empty for QONNX custom ops
        )
        graph.add_node(node)

    # 4. Graph outputs
    for out in onnx_graph.output:
        t = graph.get_tensor(out.name)
        shape, dtype = _shape_dtype_from_value_info(out)
        if t.shape is None:
            t.shape = shape
        if t.dtype is None:
            t.dtype = dtype
        graph.outputs.append(t)

    return graph


def _shape_dtype_from_value_info(value_info):
    """Pull whatever shape/dtype ONNX declared for a graph input/output.

    NOTE: this is a starting point only — per our architecture doc, we don't
    trust this blindly for internal tensors (it's often stale/incomplete).
    Real shape resolution happens in the shape_infer stage.
    """
    shape = None
    dtype = None
    ttype = value_info.type.tensor_type
    if ttype.HasField("elem_type"):
        dtype = _DTYPE_MAP.get(ttype.elem_type, f"dtype_{ttype.elem_type}")
    if ttype.HasField("shape"):
        dims = []
        for d in ttype.shape.dim:
            dims.append(d.dim_value if d.HasField("dim_value") else None)
        shape = tuple(dims)
    return shape, dtype
