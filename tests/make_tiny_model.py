"""
Generates a tiny single-Conv .onnx file for sanity-testing the pipeline
skeleton. This is NOT a real model — just enough to prove the plumbing
works before we test on the real CloudSatNet export.

Run: python tests/make_tiny_model.py
"""

import onnx
from onnx import helper, TensorProto
import os

conv_node = helper.make_node(
    "Conv", inputs=["x", "w"], outputs=["y"],
    kernel_shape=[3, 3], pads=[1, 1, 1, 1],
)

graph = helper.make_graph(
    nodes=[conv_node],
    name="tiny_conv_test",
    inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
    outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])],
    initializer=[
        helper.make_tensor("w", TensorProto.FLOAT, [4, 3, 3, 3], [0.01] * (4 * 3 * 3 * 3))
    ],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.checker.check_model(model)

out_path = os.path.join(os.path.dirname(__file__), "..", "models", "tiny_conv.onnx")
onnx.save(model, out_path)
print(f"Wrote {out_path}")
