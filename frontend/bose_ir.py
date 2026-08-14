"""
BoseIR — our internal representation of a model (our answer to what Relax
is to TVM: this is the name of OUR graph-level IR specifically).

In order to honour Netaji Subhas Chandra Bose,  greatest freedom fighters of all time, we have named our internal representaion(IR) to honour him


Three building blocks:
    Tensor  — one piece of data flowing through the graph (a name, a shape,
              a dtype, and who produces/consumes it)
    IRNode  — one operation (an op type, its input tensors, its output
              tensors, and any settings/attributes it needs)
    BoseIR  — the whole model: an ordered list of IRNodes, plus which
              tensors are graph inputs/outputs, plus constant data
"""


class Tensor:
    """One piece of data flowing through the graph."""

    def __init__(self, name, shape=None, dtype=None):
        self.name = name
        self.shape = shape          # tuple of ints (or None until shape inference runs)
        self.dtype = dtype          # e.g. "float32", "int8" (or None until resolved)
        self.producer = None        # the IRNode that creates this tensor (None = graph input or constant)
        self.consumers = []         # list of IRNodes that read this tensor

    def __repr__(self):
        return f"Tensor({self.name}, shape={self.shape}, dtype={self.dtype})"


class IRNode:
    """One operation in the graph."""

    def __init__(self, op_type, name, inputs, outputs, attrs=None, domain=""):
        self.op_type = op_type          # e.g. "Conv2D" — must be one of our 18 core ops after legalize
        self.name = name                # unique node name, mainly for error messages
        self.inputs = inputs            # list[Tensor]
        self.outputs = outputs          # list[Tensor]
        self.attrs = attrs or {}        # dict of settings, e.g. {"kernel_shape": [3, 3]}
        self.domain = domain            # "" = standard ONNX, non-empty = custom domain (e.g. QONNX)

        for t in inputs:
            t.consumers.append(self)
        for t in outputs:
            t.producer = self

    def __repr__(self):
        in_names = [t.name for t in self.inputs]
        out_names = [t.name for t in self.outputs]
        return f"IRNode({self.op_type}, name={self.name}, inputs={in_names}, outputs={out_names})"


class BoseIR:
    """The whole model: nodes in topological order, plus boundary tensors."""

    def __init__(self, name="graph"):
        self.name = name
        self.nodes = []                 # list[IRNode], kept in topological (execution) order
        self.inputs = []                # list[Tensor] — the graph's real inputs (e.g. the image)
        self.outputs = []               # list[Tensor] — the graph's final outputs
        self.tensors = {}               # name -> Tensor, the single source of truth for every tensor
        self.initializers = {}          # name -> numpy array, constant weight/bias data
        self.opset = None               # the model's ONNX opset version — needed for version-aware legalization

    def get_tensor(self, name):
        """Look up a tensor by name, or create a fresh placeholder if it's new."""
        if name not in self.tensors:
            self.tensors[name] = Tensor(name)
        return self.tensors[name]

    def add_node(self, node):
        self.nodes.append(node)
        for t in node.outputs:
            self.tensors[t.name] = t

    def replace_node(self, old_node, new_nodes):
        """Swap one node for one-or-more replacement nodes, at the same position."""
        idx = self.nodes.index(old_node)
        self.nodes[idx:idx + 1] = new_nodes
        for t in old_node.outputs:
            for c in list(t.consumers):
                if c is old_node:
                    t.consumers.remove(c)

    def remove_node(self, node):
        self.nodes.remove(node)

    def __repr__(self):
        return f"BoseIR({self.name}, {len(self.nodes)} nodes, {len(self.inputs)} inputs, {len(self.outputs)} outputs)"
