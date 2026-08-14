"""
Generates a Conv -> BatchNorm -> Relu test model — this is the pattern
_fold_batchnorm() needs to handle correctly.

Run: python tests/make_convbn_model.py
"""

import onnx
from onnx import helper, TensorProto
import numpy as np
import os

np.random.seed(0)

conv = helper.make_node("Conv", inputs=["x", "w"], outputs=["y"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1])
bn = helper.make_node("BatchNormalization", inputs=["y", "gamma", "beta", "mean", "var"],
                       outputs=["y_bn"], epsilon=1e-5)
relu = helper.make_node("Relu", inputs=["y_bn"], outputs=["z"])

C_out = 4
W = np.random.randn(C_out, 3, 3, 3).astype(np.float32) * 0.1
gamma = np.random.uniform(0.5, 1.5, C_out).astype(np.float32)
beta = np.random.uniform(-0.5, 0.5, C_out).astype(np.float32)
mean = np.random.uniform(-0.1, 0.1, C_out).astype(np.float32)
var = np.random.uniform(0.5, 1.5, C_out).astype(np.float32)

graph = helper.make_graph(
    nodes=[conv, bn, relu],
    name="convbn_test",
    inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
    outputs=[helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, C_out, 8, 8])],
    initializer=[
        helper.make_tensor("w", TensorProto.FLOAT, W.shape, W.flatten()),
        helper.make_tensor("gamma", TensorProto.FLOAT, gamma.shape, gamma),
        helper.make_tensor("beta", TensorProto.FLOAT, beta.shape, beta),
        helper.make_tensor("mean", TensorProto.FLOAT, mean.shape, mean),
        helper.make_tensor("var", TensorProto.FLOAT, var.shape, var),
    ],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.checker.check_model(model)

out_path = os.path.join(os.path.dirname(__file__), "..", "models", "convbn.onnx")
onnx.save(model, out_path)
print(f"Wrote {out_path}")

# Also save the reference numpy arrays so we can numerically check the fold
np.savez(os.path.join(os.path.dirname(__file__), "..", "models", "convbn_ref.npz"),
         x=np.random.randn(1, 3, 8, 8).astype(np.float32),
         W=W, gamma=gamma, beta=beta, mean=mean, var=var)
