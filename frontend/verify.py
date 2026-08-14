"""
Stage 6: IR Verification

The last gate before handing the graph to the middle-end. Checks:
    1. every node's op is one of our 18 core ops (nothing slipped through
       legalize with the wrong op type — defense in depth)
    2. the graph is in a valid topological order (every node's inputs are
       produced earlier, or are graph inputs/constants)
    3. every tensor has a resolved shape and dtype
    4. no orphaned nodes (every output either feeds something else, or is
       a declared graph output)
"""

CORE_OPS = {
    "Conv2D", "MatMul", "Add", "Mul", "Max", "Min", "ReLU", "Clip",
    "MaxPool", "AveragePool", "GlobalAveragePool", "Reshape", "Concat",
    "Slice", "Transpose", "QuantizeLinear", "DequantizeLinear",
    "Requantize", "Constant", "Identity",
}


def verify(graph):
    _check_only_core_ops(graph)
    _check_topological_order(graph)
    _check_shapes_and_dtypes_resolved(graph)
    _check_no_orphans(graph)
    return True


def _check_only_core_ops(graph):
    for node in graph.nodes:
        if node.op_type not in CORE_OPS:
            raise ValueError(
                f"IR VERIFICATION FAILED: node '{node.name}' has op_type '{node.op_type}', "
                f"which is not one of our 18 core ops. This means legalize() has a bug — "
                f"it should never have let this through."
            )


def _check_topological_order(graph):
    produced = set()
    for t in graph.inputs:
        produced.add(t.name)
    for t in graph.tensors.values():
        if t.producer is None:
            produced.add(t.name)  # graph input or constant with no explicit producer node

    for node in graph.nodes:
        for t in node.inputs:
            if t.name not in produced:
                raise ValueError(
                    f"IR VERIFICATION FAILED: node '{node.name}' reads tensor '{t.name}' "
                    f"before it's produced. BoseIR is not in valid topological order."
                )
        for t in node.outputs:
            produced.add(t.name)


def _check_shapes_and_dtypes_resolved(graph):
    for node in graph.nodes:
        for t in node.outputs:
            if t.shape is None:
                raise ValueError(f"IR VERIFICATION FAILED: tensor '{t.name}' (from node '{node.name}') "
                                  f"has no resolved shape — shape inference missed it.")
            if t.dtype is None:
                raise ValueError(f"IR VERIFICATION FAILED: tensor '{t.name}' (from node '{node.name}') "
                                  f"has no resolved dtype.")


def _check_no_orphans(graph):
    graph_output_names = {t.name for t in graph.outputs}
    for node in graph.nodes:
        for t in node.outputs:
            if len(t.consumers) == 0 and t.name not in graph_output_names:
                raise ValueError(
                    f"IR VERIFICATION FAILED: tensor '{t.name}' (from node '{node.name}') "
                    f"has no consumers and isn't a declared graph output — dead/orphaned node."
                )
