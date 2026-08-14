"""
Clip under an OLD opset (6) — min/max are node ATTRIBUTES.
Tests the _convert_clip_v1 path.

Run: python tests/make_clip_old_opset_model.py
"""
import onnx
from onnx import helper, TensorProto
import os

clip_node = helper.make_node("Clip", inputs=["x"], outputs=["y"], min=0.0, max=6.0)

graph = helper.make_graph(
    nodes=[clip_node], name="clip_old_opset_test",
    inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
    outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 6)])
onnx.checker.check_model(model)

out_path = os.path.join(os.path.dirname(__file__), "..", "models", "clip_old_opset.onnx")
onnx.save(model, out_path)
print(f"Wrote {out_path}")
