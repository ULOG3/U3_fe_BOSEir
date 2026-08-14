# ULOG3 DL Compiler — Frontend
![image alt](https://github.com/ULOG3/U3_fe_BOSEir/blob/dev-v0.1/boseir.png?raw=true)

Converts a `.onnx` model into BoseIR — our internal graph-level IR (the
same role Relax plays for TVM), using only our chip's
18 supported core operations.

## Setup

```bash
pip install -r requirements.txt
```

## Try it

```bash
python tests/make_tiny_model.py           # single-Conv sanity test
python tests/make_convbn_model.py         # Conv-BN-ReLU (tests BN folding)
python tests/make_residual_block_model.py # full CloudSatNet-style residual block
python tests/make_unsupported_op_model.py # Sigmoid (tests rejection path)

python main.py models/tiny_conv.onnx
python main.py models/convbn.onnx
python main.py models/residual_block.onnx
python main.py models/unsupported_op.onnx   # this one is SUPPOSED to fail, on purpose
```

## Structure

```
frontend/
    bose_ir.py        Stage 2  — Tensor / IRNode / BoseIR data structures
    import_stage.py   Stage 1+2 — validate the file, then read it into structural IR
    legalize.py        Stage 3  — BatchNorm folding, op-conversion table, QONNX Quant
                                    handling, hard rejection of unsupported ops
    shape_infer.py     Stage 4  — shape function for each of the 18 core ops
    quant_resolve.py   Stage 5  — validates scale/zero-point are concrete constants
    verify.py           Stage 6  — final invariant checks before handoff
    pipeline.py          wires all six stages together, stage-by-stage logging

main.py                            command-line entry point: `python main.py <file.onnx>`
tests/make_tiny_model.py            one-node sanity test (single Conv)
tests/make_convbn_model.py          tests BatchNorm folding specifically
tests/make_residual_block_model.py  tests the full CloudSatNet-style residual pattern
tests/make_unsupported_op_model.py  tests the rejection path (Sigmoid)
models/                             generated test .onnx files live here
```

## Where we are

- [x] Step 1 — project skeleton
- [x] Step 2 — BoseIR data structures (`Tensor`, `IRNode`, `BoseIR`)
- [x] Step 3 — end-to-end import working (tested on a single-Conv model)
- [x] Step 4 — op-conversion table (`CONVERT_MAP` in legalize.py)
- [x] Step 5 — unsupported ops rejected with a clear error (tested with Sigmoid)
- [x] Step 6 — BatchNorm folding, **numerically verified** against running
      Conv+BN separately (max diff ~2e-7, i.e. float32 noise, not a bug)
- [x] Step 7 — tested on a full Conv-BN-ReLU-Conv-BN-Add-ReLU residual block
      (the CloudSatNet pattern) — BatchNorm fully disappears, residual Add
      wires correctly
- [x] Step 8 — shape inference + final verification pass, both working

## Opset-aware legalization (added after comparing against TVM's real importer flow)

- `BoseIR.opset` now tracks the model's ONNX opset version, extracted during import
- `legalize.py` supports version-specific converters per op (TVM calls this the
  `_impl_vX` pattern) — `Clip` is the worked example: opset<11 (min/max as
  attributes) and opset>=11 (min/max as tensor inputs) both normalize to the
  **same** internal `{"min", "max"}` attrs, verified numerically identical
  across both real .onnx files in `tests/make_clip_old_opset_model.py` and
  `tests/make_clip_new_opset_model.py`
- **Bug caught by this testing, now fixed**: `_fold_batchnorm` was deleting
  the BatchNorm node without detaching it from its input tensors' consumer
  lists first — gamma/beta/mean/var kept a stale reference to a node that
  was no longer in the graph, which was silently hiding a real dead-node
  detection gap in `verify()`. Fixed, plus added `_prune_dead_constants()`
  so constants that get resolved into attributes (like Clip's min/max) are
  cleanly removed instead of left dangling.

## What's genuinely NOT covered yet (be aware of these before you rely on it)

- **QONNX `Quant`/`BipolarQuant` conversion is implemented but not tested
  against a real Brevitas-exported model yet** — only the standard-ONNX
  path (Conv/BN/ReLU/Add) has been tested end-to-end so far. Test this
  next against a real CloudSatNet export.
- **Gemm decomposition** handles the common case (transA/transB, optional
  bias) but explicitly rejects non-1.0 `alpha`/`beta` rather than silently
  computing wrong values — check if your exported FC layer uses those.
- **Slice** only handles the opset≥10 constant-input form, not the older
  attribute-based form.
- **MatMul shape inference** only handles the 2D×2D case (the common case
  after Gemm decomposition) — will raise `NotImplementedError` on batched
  matmul rather than compute it wrong.

## Our 18 core ops (what the frontend must reduce every model down to)

Conv2D, MatMul/Gemm, Add, Mul, Max, Min, ReLU, Clip, MaxPool, AveragePool,
GlobalAveragePool, Reshape/Flatten, Concat, Slice/Split, Transpose,
QuantizeLinear/DequantizeLinear, Requantize, Constant, Identity
