"""
Generates a tiny model containing a Sigmoid — an op we deliberately don't
support. Used to test that legalize() rejects it loudly and clearly,
instead of crashing or silently letting it through.

Run: python tests/make_unsupported_op_model.py
"""

import onnx
from onnx import helper, TensorProto
import os

sigmoid_node = helper.make_node("Sigmoid", inputs=["x"], outputs=["y"])

graph = helper.make_graph(
    nodes=[sigmoid_node],
    name="unsupported_op_test",
    inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
    outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.checker.check_model(model)

out_path = os.path.join(os.path.dirname(__file__), "..", "models", "unsupported_op.onnx")
onnx.save(model, out_path)
print(f"Wrote {out_path}")
