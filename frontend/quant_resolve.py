"""
Stage 5: Quantization Resolution

IMPORTANT BOUNDARY (established in the architecture doc): this stage
CONCRETIZES quantization parameters that already exist in the graph — the
scale/zero-point values on Requantize nodes were already pulled from the
QONNX Quant nodes' constant inputs back in legalize(). This stage does NOT
insert new Requantize nodes to fix scale mismatches between arbitrary
producer/consumer pairs — that decision depends on fusion choices made
later, so it belongs to the middle-end, not here.

What this stage actually does: validate that every Requantize node's scale/
zero-point ended up as real, concrete numbers (not left unresolved), so a
graph with broken/missing quant params gets caught here — not silently
passed on to the middle-end.
"""

import numpy as np


def resolve_quantization(graph):
    for node in graph.nodes:
        if node.op_type != "Requantize":
            continue

        scale = node.attrs.get("scale")
        zero_point = node.attrs.get("zero_point")

        if scale is None:
            raise ValueError(f"Requantize node '{node.name}' has no scale value — "
                              f"quantization wasn't fully resolved during import.")
        if not isinstance(scale, (np.ndarray, int, float)):
            raise ValueError(f"Requantize node '{node.name}': scale is not a concrete constant "
                              f"(got {type(scale)}) — dynamic/runtime scales aren't supported.")
        if zero_point is None:
            raise ValueError(f"Requantize node '{node.name}' has no zero_point value.")

    return graph
