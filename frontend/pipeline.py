"""
Pipeline — wires all six stages together in order.

If any stage fails, the error message tells you exactly which stage and
which node caused it — that's on purpose (see the architecture doc's
"fail-fast with structured diagnostics" principle).
"""

from frontend.import_stage import validate, import_model
from frontend.legalize import legalize
from frontend.shape_infer import infer_shapes
from frontend.quant_resolve import resolve_quantization
from frontend.verify import verify


def run(onnx_path: str, verbose: bool = True):
    def log(msg):
        if verbose:
            print(msg)

    log(f"\n--- Running frontend pipeline on: {onnx_path} ---\n")

    log("[Stage 1] Validate ...")
    model = validate(onnx_path)
    log("    OK — file loaded and passed onnx.checker\n")

    log("[Stage 2] Import (protobuf -> structural IR) ...")
    graph = import_model(model)
    log(f"    OK — {len(graph.nodes)} nodes imported\n")

    log("[Stage 3] Legalize (-> our 18 core ops) ...")
    graph = legalize(graph)
    log(f"    OK — {len(graph.nodes)} nodes after legalization\n")

    log("[Stage 4] Shape & Layout Inference ...")
    graph = infer_shapes(graph)
    log("    OK — every tensor shape resolved\n")

    log("[Stage 5] Quantization Resolution ...")
    graph = resolve_quantization(graph)
    log("    OK\n")

    log("[Stage 6] IR Verification ...")
    verify(graph)
    log("    OK — graph verified clean\n")

    log("--- Pipeline complete. BoseIR is valid and ready for the middle-end. ---\n")
    return graph
