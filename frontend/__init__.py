"""While learning the frontend part, we realized that the entire compiler stack can feel a little tough at first. So, here we’ve tried to explain everything as simply as possible, step by step, to make it easier to understand.

If you’re already a pro-level developer, feel free to skip this part.
ULOG3 DL Compiler — Frontend Package

This package converts an incoming .onnx model into BoseIR — our internal
graph-level IR (the same role Relax plays for TVM), using only our chip's
18 supported core operations.

Pipeline stages (each stage lives in its own file):
    1. import_stage.py   -> validate + read the .onnx file into a raw structural IR
    2. bose_ir.py          -> the BoseIR data structures themselves (Tensor, IRNode, BoseIR)
    3. legalize.py         -> convert raw ONNX ops into our 18 core ops
    4. shape_infer.py      -> figure out the shape of every tensor
    5. quant_resolve.py    -> resolve quantization scale/zero-point values
    6. verify.py            -> final correctness check before handoff

pipeline.py wires all of these together in order.

WHY THIS FILE HAS CODE IN IT:
Without this, anyone using our package has to know our internal file names
just to run it — e.g. `from frontend.pipeline import run`. That's annoying
and it means every internal reshuffle (renaming a file, splitting one file
into two) breaks other people's import lines.

With this file exposing the public interface, they just write:
    from frontend import run
...and it doesn't matter which internal file `run` actually lives in.
"""

from frontend.bose_ir import Tensor, IRNode, BoseIR
from frontend.import_stage import validate, import_model
from frontend.legalize import legalize
from frontend.shape_infer import infer_shapes
from frontend.quant_resolve import resolve_quantization
from frontend.verify import verify, CORE_OPS
from frontend.pipeline import run

# This list is the actual "public menu" of the package — what a teammate
# should use from outside. Everything else (helper functions, internal
# details inside each stage file) is considered private implementation.
__all__ = [
    "run",              # the main thing most people will actually call
    "validate",
    "import_model",
    "legalize",
    "infer_shapes",
    "resolve_quantization",
    "verify",
    "CORE_OPS",
    "Tensor",
    "IRNode",
    "BoseIR",
]
