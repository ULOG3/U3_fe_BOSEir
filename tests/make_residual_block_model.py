"""
Generates a Conv-BN-ReLU-Conv-BN-Add(skip)-ReLU block — the exact pattern
from the CloudSatNet worked trace in the architecture doc. This is the
strongest end-to-end test: two BN-folds plus a residual Add in one graph.

Run: python tests/make_residual_block_model.py
"""

import onnx
from onnx import helper, TensorProto
import numpy as np
import os

np.random.seed(1)
C = 4

def bn_inits(prefix):
    return (
        np.random.uniform(0.5, 1.5, C).astype(np.float32),
        np.random.uniform(-0.5, 0.5, C).astype(np.float32),
        np.random.uniform(-0.1, 0.1, C).astype(np.float32),
        np.random.uniform(0.5, 1.5, C).astype(np.float32),
    )

W1 = (np.random.randn(C, C, 3, 3) * 0.1).astype(np.float32)
W2 = (np.random.randn(C, C, 3, 3) * 0.1).astype(np.float32)
g1, b1, m1, v1 = bn_inits("bn1")
g2, b2, m2, v2 = bn_inits("bn2")

nodes = [
    helper.make_node("Conv", ["x", "w1"], ["y1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node("BatchNormalization", ["y1", "g1", "b1", "m1", "v1"], ["y1_bn"], epsilon=1e-5),
    helper.make_node("Relu", ["y1_bn"], ["z1"]),
    helper.make_node("Conv", ["z1", "w2"], ["y2"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node("BatchNormalization", ["y2", "g2", "b2", "m2", "v2"], ["y2_bn"], epsilon=1e-5),
    helper.make_node("Add", ["y2_bn", "x"], ["res"]),
    helper.make_node("Relu", ["res"], ["out"]),
]

def T(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.flatten())

graph = helper.make_graph(
    nodes=nodes,
    name="residual_block_test",
    inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, 8, 8])],
    outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, C, 8, 8])],
    initializer=[
        T("w1", W1), T("g1", g1), T("b1", b1), T("m1", m1), T("v1", v1),
        T("w2", W2), T("g2", g2), T("b2", b2), T("m2", m2), T("v2", v2),
    ],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.checker.check_model(model)

out_path = os.path.join(os.path.dirname(__file__), "..", "models", "residual_block.onnx")
onnx.save(model, out_path)
print(f"Wrote {out_path}")
