"""CPU oracle checks for offline graph rewrites (development dependencies only)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h
from onnx import numpy_helper as nh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export import Exporter, fold_parallel_convolutions


class Folding(unittest.TestCase):
    def model(self, extra_consumer=False, stride=1):
        rng = np.random.default_rng(18)
        initializers = [
            nh.from_array(rng.normal(size=(4, 3, *k)).astype("f"), f"w{i}")
            for i, k in enumerate(((3, 3), (3, 1), (1, 3)))
        ]
        initializers.append(nh.from_array(np.arange(4, dtype="f"), "bias"))
        nodes = [
            h.make_node("Conv", ["x", f"w{i}", "bias"], [f"c{i}"], pads=p, strides=[stride, stride])
            for i, p in enumerate(([1, 1, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]))
        ]
        nodes += [h.make_node("Add", ["c0", "c1"], ["a"]), h.make_node("Add", ["a", "c2"], ["y"])]
        outs = [h.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4, 7, 9])]
        if extra_consumer:
            outs.append(h.make_tensor_value_info("c0", onnx.TensorProto.FLOAT, [1, 4, 7, 9]))
        model = h.make_model(
            h.make_graph(
                nodes,
                "fold",
                [h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 7, 9])],
                outs,
                initializers,
            ),
            opset_imports=[h.make_opsetid("", 17)],
        )
        model.ir_version = 10
        return model

    def test_equivalence_and_shared_bias(self):
        m = self.model()
        x = np.random.default_rng(4).normal(size=(1, 3, 7, 9)).astype("f")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        before = ort.InferenceSession(m.SerializeToString(), options).run(None, {"x": x})[0]
        self.assertEqual(fold_parallel_convolutions(m), 2)
        onnx.checker.check_model(m)
        after = ort.InferenceSession(m.SerializeToString(), options).run(None, {"x": x})[0]
        np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-5)
        self.assertEqual(len(m.graph.node), 1)

    def test_external_consumer_prevents_rewrite(self):
        self.assertEqual(fold_parallel_convolutions(self.model(extra_consumer=True)), 0)

    def test_stride_is_not_rewritten(self):
        self.assertEqual(fold_parallel_convolutions(self.model(stride=2)), 0)

    def test_transpose_roundtrip_across_view(self):
        m = h.make_model(
            h.make_graph(
                [], "layout", [], [h.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 5, 3])]
            )
        )
        e = Exporter(m, 1, 5, height=1)
        e.shapes["x"] = (1, 1, 5, 3)
        e.emit("transpose", ["x"], "mid", (1, 3, 1, 5), "{1,1,5,3},{1,3,1,5},{0,3,1,2}")
        e.emit("transpose", ["mid"], "y", (1, 5, 3), "{1,3,5},{1,5,3},{0,2,1}")
        e.eliminate_inverse_transposes()
        self.assertEqual(e.ops, [])
        self.assertEqual(e.tensor("y"), e.tensor("x"))


if __name__ == "__main__":
    unittest.main()
