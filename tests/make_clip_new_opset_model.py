"""
Clip under a NEW opset (13) — min/max are TENSOR INPUTS.
Tests the _convert_clip_v11 path. Same 0.0/6.0 bounds as the old-opset
version — after legalize, both should normalize to identical attrs.

Run: python tests/make_clip_new_opset_model.py
"""
import onnx
from onnx import helper, TensorProto
import os

clip_node = helper.make_node("Clip", inputs=["x", "min_t", "max_t"], outputs=["y"])

graph = helper.make_graph(
    nodes=[clip_node], name="clip_new_opset_test",
    inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
    outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    initializer=[
        helper.make_tensor("min_t", TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor("max_t", TensorProto.FLOAT, [], [6.0]),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.checker.check_model(model)

out_path = os.path.join(os.path.dirname(__file__), "..", "models", "clip_new_opset.onnx")
onnx.save(model, out_path)
print(f"Wrote {out_path}")
