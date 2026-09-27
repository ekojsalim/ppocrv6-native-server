#!/usr/bin/env python3
"""Offline, fixed-shape PP-OCRv6 lowering to native calls and raw FP32 weights.

ONNX is an authoring/validation dependency only. The generated executable does
not parse a graph or link an inference framework. Unsupported operators fail.
"""

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def fold_parallel_convolutions(model):
    """Fold single-use, aligned linear Conv+Conv branches before FP16 packing.

    Padding offsets embed each filter in the larger receptive field. Biases
    are summed too, even when the original branches share a bias initializer.
    Non-unit dilation/stride and automatic padding are deliberately excluded.
    """
    constants = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    nodes = list(model.graph.node)
    folded = 0
    while True:
        producers = {out: n for n in nodes for out in n.output}
        uses = collections.Counter(i for n in nodes for i in n.input)
        uses.update(o.name for o in model.graph.output)
        replacement = None
        for node in nodes:
            if node.op_type != "Add" or len(node.input) != 2:
                continue
            pair = [producers.get(i) for i in node.input]
            if any(n is None or n.op_type != "Conv" or uses[n.output[0]] != 1 for n in pair):
                continue
            a, b = pair
            if a.input[0] != b.input[0] or any(
                i not in constants for n in pair for i in n.input[1:]
            ):
                continue
            attrs = [{x.name: helper.get_attribute_value(x) for x in n.attribute} for n in pair]
            if any(
                x.get("auto_pad", b"NOTSET") != b"NOTSET"
                or x.get("strides", [1, 1]) != [1, 1]
                or x.get("dilations", [1, 1]) != [1, 1]
                for x in attrs
            ):
                continue
            if attrs[0].get("group", 1) != attrs[1].get("group", 1):
                continue
            weights = [constants[n.input[1]] for n in pair]
            if weights[0].shape[:2] != weights[1].shape[:2]:
                continue
            pads = [x.get("pads", [0] * 4) for x in attrs]
            if any(
                p[:2] != p[2:] or list(w.shape[2:]) != [2 * p[0] + 1, 2 * p[1] + 1]
                for p, w in zip(pads, weights)
            ):
                continue
            ph, pw = (max(p[j] for p in pads) for j in range(2))
            weight = np.zeros((*weights[0].shape[:2], 2 * ph + 1, 2 * pw + 1), np.float32)
            bias = np.zeros(weights[0].shape[0], np.float32)
            for n, w, p in zip(pair, weights, pads):
                y, x = ph - p[0], pw - p[1]
                weight[:, :, y : y + w.shape[2], x : x + w.shape[3]] += w
                if len(n.input) == 3:
                    bias += constants[n.input[2]]
            prefix = node.output[0] + "_folded"
            for name, value in [(prefix + "_w", weight), (prefix + "_b", bias)]:
                constants[name] = value
                model.graph.initializer.append(numpy_helper.from_array(value, name))
            conv = helper.make_node(
                "Conv",
                [a.input[0], prefix + "_w", prefix + "_b"],
                list(node.output),
                name=prefix,
                group=attrs[0].get("group", 1),
                kernel_shape=[2 * ph + 1, 2 * pw + 1],
                pads=[ph, pw, ph, pw],
                strides=[1, 1],
            )
            replacement = (node, pair, conv)
            break
        if replacement is None:
            break
        add, pair, conv = replacement
        nodes = [conv if n is add else n for n in nodes if all(n is not branch for branch in pair)]
        folded += 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {i for n in nodes for i in n.input} | {o.name for o in model.graph.output}
    kept = [v for v in model.graph.initializer if v.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return folded


def vec(v):
    return "{" + ",".join(str(int(x)) for x in v) + "}"


def flat_broadcast(source, target):
    """Return constant divisor/modulus for contiguous broadcast indexing."""
    source = (1,) * (len(target) - len(source)) + tuple(source)
    if math.prod(source) == math.prod(target):
        return (1, 0)
    axes = [i for i, size in enumerate(source) if size != 1]
    if not axes:
        return (1, 1)
    if any(source[i] != target[i] for i in range(axes[0], axes[-1] + 1)):
        return None
    divisor = math.prod(target[axes[-1] + 1 :])
    modulus = math.prod(source)
    return divisor, 0 if divisor * modulus == math.prod(target) else modulus


def broadcast_expression(source, target):
    index = flat_broadcast(source, target)
    if index is not None:
        return f"flat_index<{index[0]},{index[1]}>(i)"
    source = (1,) * (len(target) - len(source)) + tuple(source)
    terms = []
    for j, size in enumerate(source):
        if size == 1:
            continue
        dest_stride = math.prod(target[j + 1 :])
        source_stride = math.prod(source[j + 1 :])
        value = "i" if dest_stride == 1 else f"(i/{dest_stride})"
        if j:
            value = f"({value}%{target[j]})"
        if source_stride != 1:
            value = f"({value}*{source_stride})"
        terms.append(value)
    return "(" + "+".join(terms) + ")"


class Exporter:
    def __init__(self, model, batch, width, optimize=False, fuse=False, height=48):
        self.model = model
        self.shapes = {"x": (batch, 3, height, width)}
        self.constants = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
        self.shapes.update({k: v.shape for k, v in self.constants.items()})
        self.alias = {}
        self.ids = {}
        self.tensors = []
        self.ops = []
        self.optimize = optimize
        self.fuse = fuse
        self.specs = {}
        self.kernels = []
        self.layernorm_fusions = 0
        self.pointwise_groups = {}
        self.gemm_epilogues = collections.Counter()
        self.tensor("x")

    def root(self, name):
        while name in self.alias:
            name = self.alias[name]
        return name

    def tensor(self, name):
        name = self.root(name)
        if name not in self.ids:
            self.ids[name] = len(self.tensors)
            self.tensors.append(name)
        return self.ids[name]

    def emit(self, kind, inputs, output, shape, args=""):
        self.shapes[output] = tuple(shape)
        ii = [self.tensor(i) for i in inputs]
        oi = self.tensor(output)
        self.specs[oi] = {
            "kind": kind,
            "args": args,
            "input_shapes": [self.shapes[i] for i in inputs],
        }
        if self.optimize and kind == "binary":
            indices = [flat_broadcast(self.shapes[i], shape) for i in inputs]
            if all(index is not None for index in indices):
                mode = int(args.rsplit(",", 1)[-1])
                kind = f"binary_fast<{mode},{indices[0][0]},{indices[0][1]},{indices[1][0]},{indices[1][1]}>"
                args = str(math.prod(shape))
        params = ",".join(map(str, ii + [oi]))
        self.ops.append((ii, oi, f"r.{kind}({params}{',' if args else ''}{args});", output))

    def lower(self):
        for node in self.model.graph.node:
            op = node.op_type
            a = {x.name: helper.get_attribute_value(x) for x in node.attribute}
            ins = [self.root(x) for x in node.input]
            out = node.output[0]
            s = self.shapes[ins[0]] if ins else ()
            c = self.constants
            if op == "Identity":
                self.alias[out] = ins[0]
                self.shapes[out] = s
                continue
            if op == "Shape":
                c[out] = np.array(s, dtype=np.int64)
            elif op == "Constant":
                c[out] = numpy_helper.to_array(a["value"])
            elif op in ("Reshape", "Squeeze", "Unsqueeze"):
                if op == "Reshape":
                    target = [s[j] if v == 0 else int(v) for j, v in enumerate(c[ins[1]])]
                    if -1 in target:
                        target[target.index(-1)] = math.prod(s) // -math.prod(target)
                elif op == "Squeeze":
                    axes = [
                        int(x) % len(s)
                        for x in a.get("axes", c.get(ins[1] if len(ins) > 1 else "", []))
                    ]
                    assert all(s[x] == 1 for x in axes)
                    target = [v for j, v in enumerate(s) if j not in axes]
                else:
                    axes = a.get("axes", c.get(ins[1] if len(ins) > 1 else "", []))
                    target = list(s)
                    for axis in sorted(int(x) % (len(s) + len(axes)) for x in axes):
                        target.insert(axis, 1)
                assert math.prod(target) == math.prod(s)
                self.shapes[out] = tuple(target)
                if ins[0] in c:
                    c[out] = c[ins[0]].reshape(target)
                else:
                    # Views have their own shape but share their storage ID.
                    self.ids[out] = self.tensor(ins[0])
                continue
            elif op == "Slice":
                slices = [slice(None)] * len(s)
                axes = c[ins[3]] if len(ins) > 3 else range(len(c[ins[1]]))
                steps = c[ins[4]] if len(ins) > 4 else [1] * len(axes)
                for axis, start, end, step in zip(axes, c[ins[1]], c[ins[2]], steps):
                    slices[int(axis)] = slice(int(start), int(end), int(step))
                bounds = [sl.indices(d) for sl, d in zip(slices, s)]
                target = [len(range(*b)) for b in bounds]
                if ins[0] in c:
                    c[out] = c[ins[0]][tuple(slices)]
                else:
                    assert all(b[2] == 1 for b in bounds), "Only unit slices supported"
                    self.emit(
                        "slice",
                        ins[:1],
                        out,
                        target,
                        f"{vec(s)},{vec(target)},{vec([b[0] for b in bounds])}",
                    )
            elif op == "Concat":
                axis = a["axis"] % len(s)
                if all(i in c for i in ins):
                    c[out] = np.concatenate([c[i] for i in ins], axis=axis)
                else:
                    target = list(s)
                    target[axis] = sum(self.shapes[i][axis] for i in ins)
                    current = ins[0]
                    for j, other in enumerate(ins[1:], 1):
                        left = self.shapes[current]
                        right = self.shapes[other]
                        target = list(left)
                        target[axis] += right[axis]
                        name = out if j == len(ins) - 1 else out + f"_concat_{j}"
                        self.emit(
                            "concat",
                            [current, other],
                            name,
                            target,
                            f"{math.prod(left[:axis])},{left[axis] * math.prod(left[axis + 1 :])},"
                            f"{right[axis] * math.prod(left[axis + 1 :])}",
                        )
                        current = name
            elif all(i in c for i in ins):
                xs = [c[i] for i in ins]
                funcs = {
                    "Add": np.add,
                    "Sub": np.subtract,
                    "Mul": np.multiply,
                    "Div": np.divide,
                    "Pow": np.power,
                    "Sqrt": np.sqrt,
                }
                if op == "Cast":
                    c[out] = xs[0].astype(helper.tensor_dtype_to_np_dtype(a["to"]))
                elif op in funcs:
                    c[out] = funcs[op](*xs)
                else:
                    raise ValueError(f"Unsupported constant op {op}")
            elif op == "Conv":
                w = self.shapes[ins[1]]
                stride = a.get("strides", [1, 1])
                dilation = a.get("dilations", [1, 1])
                pads = a.get("pads", [0, 0, 0, 0])
                if a.get("auto_pad") == b"SAME_UPPER":
                    totals = [
                        max(
                            0,
                            (math.ceil(s[j + 2] / stride[j]) - 1) * stride[j]
                            + (w[j + 2] - 1) * dilation[j]
                            + 1
                            - s[j + 2],
                        )
                        for j in range(2)
                    ]
                    pads = [v // 2 for v in totals] + [v - v // 2 for v in totals]
                if pads[:2] != pads[2:]:
                    padded = out + "_pad"
                    target = (*s[:2], s[2] + pads[0] + pads[2], s[3] + pads[1] + pads[3])
                    self.emit(
                        "pad",
                        ins[:1],
                        padded,
                        target,
                        f"{vec(s)},{vec(target)},{pads[0]},{pads[1]}",
                    )
                    ins[0] = padded
                    s = target
                    pads = [0] * 4
                target = (
                    *s[:1],
                    w[0],
                    *[
                        (s[j + 2] + pads[j] + pads[j + 2] - dilation[j] * (w[j + 2] - 1) - 1)
                        // stride[j]
                        + 1
                        for j in range(2)
                    ],
                )
                conv_out = out if len(ins) == 2 else out + "_before_bias"
                self.emit(
                    "conv",
                    ins[:2],
                    conv_out,
                    target,
                    f"{vec(s)},{vec(w)},{vec(target)},{vec(pads[:2])},{vec(stride)},{vec(dilation)},{a.get('group', 1)}",
                )
                if len(ins) == 3:
                    bias = out + "_bias"
                    c[bias] = c[ins[2]].reshape(1, -1, 1, 1)
                    self.shapes[bias] = c[bias].shape
                    self.emit(
                        "binary",
                        [conv_out, bias],
                        out,
                        target,
                        f"{vec(target)},{vec(c[bias].shape)},{vec(target)},0",
                    )
            elif op == "Resize":
                if (
                    a.get("mode") != b"nearest"
                    or a.get("coordinate_transformation_mode") != b"asymmetric"
                    or a.get("nearest_mode") != b"floor"
                ):
                    raise ValueError("Only asymmetric/floor nearest Resize is supported")
                scales = np.asarray(c[ins[2]])
                if (
                    len(s) != 4
                    or len(scales) != 4
                    or not np.all(scales[:2] == 1)
                    or np.any(scales <= 0)
                ):
                    raise ValueError("Unsupported Resize scales")
                target = tuple(math.floor(d * v) for d, v in zip(s, scales))
                self.emit(
                    "resize_nearest",
                    ins[:1],
                    out,
                    target,
                    f"{vec(s)},{vec(target)},{float(scales[2])}f,{float(scales[3])}f",
                )
            elif op == "ConvTranspose":
                w = self.shapes[ins[1]]
                stride = a.get("strides", [1, 1])
                pads = a.get("pads", [0] * 4)
                dilation = a.get("dilations", [1, 1])
                if (
                    len(ins) != 2
                    or a.get("group", 1) != 1
                    or pads[:2] != pads[2:]
                    or any(a.get("output_padding", [0, 0]))
                    or "output_shape" in a
                ):
                    raise ValueError("Unsupported transposed convolution attributes")
                target = (
                    s[0],
                    w[1],
                    *(
                        (s[j + 2] - 1) * stride[j]
                        - pads[j]
                        - pads[j + 2]
                        + dilation[j] * (w[j + 2] - 1)
                        + 1
                        for j in range(2)
                    ),
                )
                self.emit(
                    "deconv",
                    ins,
                    out,
                    target,
                    f"{vec(s)},{vec(w)},{vec(target)},{vec(pads[:2])},{vec(stride)},{vec(dilation)}",
                )
            elif op in ("MaxPool", "AveragePool"):
                k = a["kernel_shape"]
                stride = a.get("strides", [1, 1])
                pads = a.get("pads", [0] * 4)
                assert not a.get("ceil_mode", 0) and not a.get("count_include_pad", 0)
                if a.get("auto_pad") == b"SAME_UPPER":
                    totals = [
                        max(0, (math.ceil(s[j + 2] / stride[j]) - 1) * stride[j] + k[j] - s[j + 2])
                        for j in range(2)
                    ]
                    pads = [v // 2 for v in totals] + [v - v // 2 for v in totals]
                target = (
                    *s[:2],
                    *[(s[j + 2] + pads[j] + pads[j + 2] - k[j]) // stride[j] + 1 for j in range(2)],
                )
                self.emit(
                    "pool",
                    ins,
                    out,
                    target,
                    f"{vec(s)},{vec(target)},{vec(k)},{vec(stride)},{vec(pads[:2])},{str(op == 'MaxPool').lower()}",
                )
            elif op in ("Add", "Sub", "Mul", "Div", "Pow"):
                t = self.shapes[ins[1]]
                target = np.broadcast_shapes(s, t)
                self.emit(
                    "binary",
                    ins,
                    out,
                    target,
                    f"{vec(s)},{vec(t)},{vec(target)},{['Add', 'Sub', 'Mul', 'Div', 'Pow'].index(op)}",
                )
            elif op in ("Relu", "Sigmoid", "HardSigmoid", "Erf", "Sqrt", "Cast"):
                if op == "Cast":
                    assert a["to"] == onnx.TensorProto.FLOAT16
                mode = ["Relu", "Sigmoid", "HardSigmoid", "Erf", "Sqrt", "Cast"].index(op)
                self.emit(
                    "unary",
                    ins,
                    out,
                    s,
                    f"{math.prod(s)},{mode},{a.get('alpha', 0.2):.12g}f,{a.get('beta', 0.5):.12g}f",
                )
            elif op == "ReduceMean":
                axes = sorted(x % len(s) for x in a["axes"])
                assert axes == list(range(axes[0], len(s))), "Only suffix reductions supported"
                target = list(s[: axes[0]]) + ([1] * len(axes) if a.get("keepdims", 1) else [])
                self.emit(
                    "mean", ins, out, target, f"{math.prod(s[: axes[0]])},{math.prod(s[axes[0] :])}"
                )
            elif op == "Transpose":
                perm = a.get("perm", list(reversed(range(len(s)))))
                target = [s[i] for i in perm]
                self.emit("transpose", ins, out, target, f"{vec(s)},{vec(target)},{vec(perm)}")
            elif op == "MatMul":
                t = self.shapes[ins[1]]
                assert s[-1] == t[-2]
                target = (*np.broadcast_shapes(s[:-2], t[:-2]), s[-2], t[-1])
                if len(t) == 2:
                    batches, m, n, k = 1, math.prod(s[:-1]), t[-1], s[-1]
                else:
                    assert s[:-2] == t[:-2], "Unsupported matrix batch broadcast"
                    batches, m, n, k = math.prod(s[:-2]), s[-2], t[-1], s[-1]
                self.emit("matmul", ins, out, target, f"{batches},{m},{n},{k}")
            elif op == "Softmax":
                assert a["axis"] % len(s) == len(s) - 1
                self.emit("softmax", ins, out, s, f"{math.prod(s[:-1])},{s[-1]}")
            elif op == "BatchNormalization":
                scale, bias, mean, var = [c[i] for i in ins[1:]]
                mul = scale / np.sqrt(var + a.get("epsilon", 1e-5))
                add = bias - mean * mul
                for suffix, v in [("scale", mul), ("bias", add)]:
                    name = out + "_" + suffix
                    c[name] = v.reshape(1, -1, 1, 1)
                    self.shapes[name] = c[name].shape
                mid = out + "_scaled"
                self.emit(
                    "binary",
                    [ins[0], out + "_scale"],
                    mid,
                    s,
                    f"{vec(s)},{vec(c[out + '_scale'].shape)},{vec(s)},2",
                )
                self.emit(
                    "binary",
                    [mid, out + "_bias"],
                    out,
                    s,
                    f"{vec(s)},{vec(c[out + '_bias'].shape)},{vec(s)},0",
                )
            else:
                raise ValueError(f"Unsupported {op}: {node.name}")
            if out in c:
                self.shapes[out] = c[out].shape

    def eliminate_inverse_transposes(self):
        """Cancel layout round trips, including views that remove singleton axes."""
        producers = {}
        aliases = {}

        def root(i):
            while i in aliases:
                i = aliases[i]
            return i

        def reduced(spec):
            xs, _ys, perm = json.loads("[" + spec["args"].replace("{", "[").replace("}", "]") + "]")
            axes = [j for j, d in enumerate(xs) if d != 1]
            return [xs[j] for j in axes], [axes.index(j) for j in perm if xs[j] != 1]

        result = []
        for inputs, out, code, name in self.ops:
            ins = [root(i) for i in inputs]
            spec = self.specs[out]
            if spec["kind"] == "transpose" and ins[0] in producers:
                previous = producers[ins[0]]
                ps = self.specs[previous[1]]
                if ps["kind"] == "transpose":
                    xs, p = reduced(ps)
                    mid, q = reduced(spec)
                    if mid == [xs[j] for j in p] and [p[j] for j in q] == list(range(len(p))):
                        aliases[out] = root(previous[0][0])
                        continue
            if ins != inputs:
                code = (
                    f"r.{spec['kind']}("
                    + ",".join(map(str, ins + [out]))
                    + ("," + spec["args"] if spec["args"] else "")
                    + ");"
                )
            op = (ins, out, code, name)
            result.append(op)
            producers[out] = op
        self.ids = {name: root(i) for name, i in self.ids.items()}
        live = {self.tensor(self.model.graph.output[0].name)}
        kept = []
        for op in reversed(result):
            if op[1] in live:
                kept.append(op)
                live.update(op[0])
        self.ops = list(reversed(kept))

    def channels_last(self):
        """Keep the convolutional trunk NHWC; materialize NCHW at unsupported boundaries."""
        logical = dict(self.shapes)
        layout = set()
        converted_weights = set()
        result = []

        def nhwc(s):
            return (s[0], s[2], s[3], s[1])

        def parse(args):
            return json.loads("[" + args.replace("{", "[").replace("}", "]") + "]")

        def record(ins, out, kind, args, name, input_shapes):
            call = (
                f"r.{kind}(" + ",".join(map(str, ins + [out])) + ("," + args if args else "") + ");"
            )
            result.append((ins, out, call, name))
            self.specs[out] = {"kind": kind, "args": args, "input_shapes": input_shapes}

        for inputs, out, code, name in self.ops:
            spec = self.specs[out]
            kind = spec["kind"]
            args = spec["args"]
            ins = list(inputs)
            shapes = spec["input_shapes"]
            ys = logical[name]
            use = False
            if kind in ("conv", "deconv"):
                xs = parse(args)[0]
                if ins[0] not in layout:
                    temp = f"{name}_input_nhwc"
                    self.shapes[temp] = nhwc(xs)
                    logical[temp] = xs
                    new = self.tensor(temp)
                    record(
                        [ins[0]],
                        new,
                        "transpose",
                        f"{vec(xs)},{vec(nhwc(xs))},{{0,2,3,1}}",
                        temp,
                        [xs],
                    )
                    layout.add(new)
                    ins[0] = new
                wi = ins[1]
                wn = self.tensors[wi]
                if wi not in converted_weights:
                    self.constants[wn] = np.ascontiguousarray(
                        self.constants[wn].transpose(0, 2, 3, 1)
                    )
                    converted_weights.add(wi)
                args += (",true," + str(ins[0] in layout).lower()) if kind == "conv" else ",true"
                use = True
            elif kind in ("binary", "unary") and len(ys) == 4 and any(i in layout for i in ins):
                use = all(len(s) == 4 for i, s in zip(ins, shapes) if i in layout)
                if use:
                    shapes = [nhwc(s) if len(s) == 4 else s for s in shapes]
                    if kind == "binary":
                        args = (
                            f"{vec(shapes[0])},{vec(shapes[1])},{vec(nhwc(ys))},"
                            + args.rsplit(",", 1)[-1]
                        )
            elif kind == "mean" and ins[0] in layout and len(shapes[0]) == 4:
                xs = shapes[0]
                rows, cols = parse(args)
                if (
                    rows == xs[0] * xs[1]
                    and cols == xs[2] * xs[3]
                    and tuple(ys) == (xs[0], xs[1], 1, 1)
                ):
                    kind = "mean_nhwc"
                    args = f"{xs[0]},{xs[1]},{cols}"
                    use = True
            elif kind == "resize_nearest" and ins[0] in layout:
                kind = "resize_nearest_nhwc"
                use = True
            elif kind in ("pool", "pad") and ins[0] in layout:
                kind += "_nhwc"
                use = True
            elif kind == "concat" and all(i in layout for i in ins) and len(ys) == 4:
                rows, _na, _nb = parse(args)
                a, b = shapes
                if rows == ys[0] and a[2:] == b[2:] and ys[1] == a[1] + b[1]:
                    args = f"{ys[0] * ys[2] * ys[3]},{a[1]},{b[1]}"
                    use = True
            if use:
                # Residual branches can re-enter from a sequence operation in
                # NCHW. Convert those operands before interpreting them as NHWC.
                if kind in ("binary", "unary"):
                    for k, (i, s) in enumerate(zip(ins, spec["input_shapes"])):
                        if i not in layout and len(s) == 4 and s[1] > 1 and s[2] * s[3] > 1:
                            temp = f"{name}_nhwc_{k}"
                            self.shapes[temp] = nhwc(s)
                            logical[temp] = s
                            new = self.tensor(temp)
                            record(
                                [i],
                                new,
                                "transpose",
                                f"{vec(s)},{vec(nhwc(s))},{{0,2,3,1}}",
                                temp,
                                [s],
                            )
                            layout.add(new)
                            ins[k] = new
                self.shapes[name] = nhwc(ys)
                layout.add(out)
                if kind == "binary" and self.optimize:
                    indices = [flat_broadcast(s, nhwc(ys)) for s in shapes]
                    if all(ix is not None for ix in indices):
                        op = int(args.rsplit(",", 1)[-1])
                        a, b = indices
                        code = (
                            f"r.binary_fast<{op},{a[0]},{a[1]},{b[0]},{b[1]}>("
                            + ",".join(map(str, ins + [out]))
                            + f",{math.prod(ys)});"
                        )
                        result.append((ins, out, code, name))
                        self.specs[out] = {"kind": "binary", "args": args, "input_shapes": shapes}
                        continue
                record(ins, out, kind, args, name, shapes)
                continue
            # Views share storage IDs. Reconstruct the original producer layout
            # before interpreting any such view in an unsupported operator.
            for k, i in enumerate(ins):
                if i not in layout:
                    continue
                source = self.tensors[i]
                xs = logical[source]
                temp = f"{name}_nchw_{k}"
                self.shapes[temp] = xs
                logical[temp] = xs
                new = self.tensor(temp)
                record(
                    [i],
                    new,
                    "transpose",
                    f"{vec(nhwc(xs))},{vec(xs)},{{0,3,1,2}}",
                    temp,
                    [nhwc(xs)],
                )
                ins[k] = new
            if ins != inputs:
                record(ins, out, kind, args, name, shapes)
            else:
                result.append((inputs, out, code, name))
        output = self.model.graph.output[0].name
        oi = self.tensor(output)
        if oi in layout:
            shape = logical[output]
            name = output + "_output_nchw"
            self.shapes[name] = shape
            new = self.tensor(name)
            record(
                [oi],
                new,
                "transpose",
                f"{vec(nhwc(shape))},{vec(shape)},{{0,3,1,2}}",
                name,
                [nhwc(shape)],
            )
            self.ids[output] = new
            self.shapes[output] = shape
        self.ops = result
        self.nhwc_tensors = layout

    def fuse_pointwise(self):
        """Generate single-output kernels for safe contiguous elementwise regions.

        An intermediate is removed only if every consumer belongs to the fused
        region. Reshape aliases are accounted for through storage IDs.
        """
        uses = collections.defaultdict(list)
        for j, (inputs, _, _, _) in enumerate(self.ops):
            for i in inputs:
                uses[i].append(j)
        result = []
        start = 0
        while start < len(self.ops):

            def eligible(j):
                return self.specs[self.ops[j][1]]["kind"] in ("binary", "unary")

            end = start + 1
            shape = self.shapes[self.ops[start][3]]
            if eligible(start):
                while (
                    end < len(self.ops) and eligible(end) and self.shapes[self.ops[end][3]] == shape
                ):
                    end += 1
                while True:
                    split = next(
                        (
                            j + 1
                            for j in range(start, end - 1)
                            if any(user >= end for user in uses[self.ops[j][1]])
                        ),
                        end,
                    )
                    if split == end:
                        break
                    end = split
            if end - start < 2:
                result.append(self.ops[start])
                start += 1
                continue
            group = self.ops[start:end]
            internal = set()
            external = []
            body = []

            def literal(value):
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("Nonfinite scalar in fused expression")
                text = format(value, ".9g")
                if "." not in text and "e" not in text:
                    text += ".0"
                return text + "f"

            def argument(i, source, internal=internal, external=external, shape=shape):
                if i in internal:
                    return f"v{i}"
                name = self.tensors[i]
                if name in self.constants and self.constants[name].size == 1:
                    return literal(self.constants[name].item())
                if i not in external:
                    external.append(i)
                return f"float(p{i}[{broadcast_expression(source, shape)}])"

            for inputs, out, _, _ in group:
                spec = self.specs[out]
                values = [argument(i, s) for i, s in zip(inputs, spec["input_shapes"])]
                if spec["kind"] == "binary":
                    mode = int(spec["args"].rsplit(",", 1)[-1])
                    a, b = values
                    expression = f"({a} {'+-*/'[mode]} {b})" if mode < 4 else f"powf({a},{b})"
                else:
                    _, mode, alpha, beta = spec["args"].split(",")
                    a = values[0]
                    mode = int(mode)
                    expression = [
                        f"fmaxf({a},0.f)",
                        f"(1.f/(1.f+expf(-({a}))))",
                        f"fminf(1.f,fmaxf(0.f,({alpha})*({a})+({beta})))",
                        f"erff({a})",
                        f"sqrtf({a})",
                        f"__half2float(__float2half_rn({a}))",
                    ][mode]
                body.append(f"  float v{out} = {expression};")
                internal.add(out)
            out = group[-1][1]
            name = group[-1][3]
            n = math.prod(shape)
            kernel = f"fused_pointwise_{len(self.kernels)}"
            parameters = ",".join([f"const Scalar* p{i}" for i in external] + ["Scalar* y"])
            self.kernels.append(
                f"__global__ void {kernel}({parameters}) {{\n"
                f"  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>={n})return;\n"
                + "\n".join(body)
                + f"\n  y[i]=v{out};\n}}"
            )
            args = ",".join(f"r.t[{i}]" for i in external + [out])
            call = f"r.ops.emplace_back([&r]{{{kernel}<<<{(n + 255) // 256},256,0,r.stream>>>({args});}});"
            result.append((external, out, call, name))
            start = end
            self.pointwise_groups[out] = group
        self.ops = result

    def fuse_gemm_epilogues(self, residuals=False, depthwise=False):
        """Fuse proven 1x1 NHWC convolution/bias/activation patterns into cuBLASLt."""
        uses = collections.Counter(i for ins, _, _, _ in self.ops for i in ins)
        result = []
        j = 0

        def scalar(i, value):
            v = self.constants.get(self.tensors[i])
            return (
                v is not None and v.size == 1 and math.isclose(float(v.item()), value, rel_tol=1e-7)
            )

        def mode(op):
            spec = self.specs[op[1]]
            if spec["kind"] == "binary":
                return ("b", int(spec["args"].rsplit(",", 1)[-1]))
            if spec["kind"] == "unary":
                return ("u", int(spec["args"].split(",")[1]))
            return None

        while j < len(self.ops):
            conv = self.ops[j]
            ins, out, _, name = conv
            spec = self.specs[out]
            match = False
            dw = False
            if spec["kind"] == "conv" and j + 1 < len(self.ops) and uses[out] == 1:
                args = json.loads("[" + spec["args"].replace("{", "[").replace("}", "]") + "]")
                xs, ws, ys, pad, stride, dilation, groups, *layout = args
                match = (
                    layout == [True, True]
                    and groups == 1
                    and ws[2:] == [1, 1]
                    and pad == [0, 0]
                    and stride == [1, 1]
                    and dilation == [1, 1]
                    and xs[2:] == ys[2:]
                )
                dw = (
                    depthwise
                    and layout == [True, True]
                    and groups == xs[1] == ws[0]
                    and ws[1] == 1
                    and dilation == [1, 1]
                    and (
                        (ws[2:] == [3, 3] and stride in ([1, 1], [2, 1], [2, 2]))
                        or (ws[2:] == [9, 9] and stride == [1, 1])
                        or (ws[2:] == [1, 7] and stride == [1, 1])
                    )
                )
                match = match or dw
            if match:
                nxt = self.ops[j + 1]
                group = self.pointwise_groups.get(nxt[1], [nxt])
                first = group[0]
                bias = first[0][1] if len(first[0]) == 2 else -1
                match = (
                    mode(first) == ("b", 0)
                    and first[0][0] == out
                    and bias >= 0
                    and self.tensors[bias] in self.constants
                    and self.constants[self.tensors[bias]].size == ys[1]
                    and flat_broadcast(self.specs[first[1]]["input_shapes"][1], self.shapes[name])
                    == (1, ys[1])
                )
            if match:
                modes = [mode(g) for g in group]
                oi = [g[1] for g in group]
                ii = [g[0] for g in group]
                activation = -1
                residual = -1
                if len(group) == 1:
                    activation = 0
                elif modes == [("b", 0), ("u", 0)] and ii[1] == [oi[0]]:
                    activation = 1
                elif (
                    modes == [("b", 0), ("b", 3), ("u", 3), ("b", 0), ("b", 2), ("b", 2)]
                    and ii[1][0] == oi[0]
                    and scalar(ii[1][1], math.sqrt(2))
                    and ii[2] == [oi[1]]
                    and ii[3][0] == oi[2]
                    and scalar(ii[3][1], 1)
                    and ii[4] == [oi[0], oi[3]]
                    and ii[5][0] == oi[4]
                    and scalar(ii[5][1], 0.5)
                ):
                    activation = 2
                elif residuals and not dw and modes == [("b", 0), ("b", 0)] and oi[0] in ii[1]:
                    other = ii[1][1] if ii[1][0] == oi[0] else ii[1][0]
                    if self.shapes[self.tensors[other]] == self.shapes[name]:
                        activation = 0
                        residual = other
                if activation >= 0:
                    inputs = ins + [bias] + ([residual] if residual >= 0 else [])
                    target = nxt[1]
                    rows = xs[0] * xs[2] * xs[3]
                    call = (
                        "r.gemm_epilogue("
                        + ",".join(
                            map(str, ins + [bias, residual, target, rows, xs[1], ys[1], activation])
                        )
                        + ");"
                    )
                    if dw:
                        call = (
                            "r.conv("
                            + ",".join(map(str, ins + [target]))
                            + ","
                            + spec["args"]
                            + f",{bias},{activation});"
                        )
                    result.append((inputs, target, call, nxt[3]))
                    self.gemm_epilogues[
                        ("depthwise_" if dw else "")
                        + ["bias", "relu", "gelu"][activation]
                        + ("_residual" if residual >= 0 else "")
                    ] += 1
                    j += 2
                    continue
            result.append(conv)
            j += 1
        self.ops = result

    def specialize_broadcasts(self):
        """Constant-fold multidimensional indexing even for noncontiguous broadcasts."""
        for j, (inputs, out, code, name) in enumerate(self.ops):
            spec = self.specs[out]
            if spec["kind"] != "binary" or not code.startswith("r.binary("):
                continue
            shape = self.shapes[name]
            n = math.prod(shape)
            op = int(spec["args"].rsplit(",", 1)[-1])
            a, b = [
                f"float(p{k}[{broadcast_expression(s, shape)}])"
                for k, s in enumerate(spec["input_shapes"])
            ]
            expression = f"({a}{'+-*/'[op]}{b})" if op < 4 else f"powf({a},{b})"
            kernel = f"indexed_binary_{len(self.kernels)}"
            self.kernels.append(
                f"__global__ void {kernel}(const Scalar* p0,const Scalar* p1,Scalar* y) {{ int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<{n})y[i]={expression}; }}"
            )
            args = ",".join(f"r.t[{i}]" for i in inputs + [out])
            code = f"r.ops.emplace_back([&r]{{{kernel}<<<{(n + 255) // 256},256,0,r.stream>>>({args});}});"
            self.ops[j] = (inputs, out, code, name)

    def fuse_layernorm(self):
        """Keep normalization statistics and affine evaluation in FP32 registers."""
        uses = collections.defaultdict(list)
        for j, (inputs, _, _, _) in enumerate(self.ops):
            for i in inputs:
                uses[i].append(j)
        result = []
        j = 0
        while j < len(self.ops):
            group = self.ops[j : j + 9]
            kinds = [self.specs[o]["kind"] for _, o, _, _ in group]
            match = kinds == [
                "mean",
                "binary",
                "binary",
                "mean",
                "binary",
                "unary",
                "binary",
                "binary",
                "binary",
            ]
            if match:
                ins = [g[0] for g in group]
                outs = [g[1] for g in group]
                modes = [
                    int(self.specs[outs[k]]["args"].rsplit(",", 1)[-1]) for k in (1, 2, 4, 6, 7, 8)
                ]
                unary_mode = int(self.specs[outs[5]]["args"].split(",")[1])
                match = (
                    modes == [1, 4, 0, 3, 2, 0]
                    and unary_mode == 4
                    and ins[1] == [ins[0][0], outs[0]]
                    and ins[2][0] == outs[1]
                    and ins[3] == [outs[2]]
                    and ins[4][0] == outs[3]
                    and ins[5] == [outs[4]]
                    and ins[6] == [outs[1], outs[5]]
                    and ins[7][0] == outs[6]
                    and ins[8][0] == outs[7]
                    and all(all(user < j + 9 for user in uses[o]) for o in outs[:-1])
                )
                if match:
                    power = self.constants.get(self.tensors[ins[2][1]])
                    eps = self.constants.get(self.tensors[ins[4][1]])
                    rows, cols = map(int, self.specs[outs[0]]["args"].split(","))
                    match = (
                        power is not None
                        and power.size == 1
                        and float(power.item()) == 2
                        and eps is not None
                        and eps.size == 1
                        and math.prod(self.shapes[self.tensors[ins[7][1]]]) == cols
                        and math.prod(self.shapes[self.tensors[ins[8][1]]]) == cols
                    )
            if match:
                inputs = [ins[0][0], ins[7][1], ins[8][1]]
                out = outs[-1]
                args = (
                    ",".join(map(str, inputs + [out, rows, cols])) + f",{float(eps.item()):.12g}f"
                )
                result.append((inputs, out, f"r.layernorm({args});", group[-1][3]))
                self.specs[out] = {"kind": "layernorm", "args": "", "input_shapes": []}
                self.layernorm_fusions += 1
                j += 9
            else:
                result.append(self.ops[j])
                j += 1
        self.ops = result

    def write(self, directory, model_path):
        output_name = self.model.graph.output[0].name
        output_id = self.tensor(output_name)
        live = {0, output_id}
        last = {0: 0}
        for j, (ins, out, _, _) in enumerate(self.ops):
            live.update(ins)
            live.add(out)
            for i in ins:
                last[i] = j
            last.setdefault(out, j)
        last[output_id] = len(self.ops)
        last[0] = len(self.ops)  # Keep uploaded input intact across timed runs.
        # Greedy best-fit storage reuse. Outputs never alias an input to an op.
        active = {}
        free = []
        offsets = {}
        high = 0

        def alloc(i, step):
            nonlocal high
            for old, (off, size) in list(active.items()):
                if last[old] < step:
                    free.append((size, off))
                    del active[old]
            size = (math.prod(self.shapes[self.tensors[i]]) + 63) // 64 * 64
            choices = sorted((sz, off, j) for j, (sz, off) in enumerate(free) if sz >= size)
            if choices:
                sz, off, j = choices[0]
                free.pop(j)
                if sz > size:
                    free.append((sz - size, off + size))
            else:
                off = high
                high += size
            offsets[i] = off
            active[i] = (off, size)

        alloc(0, -1)
        for step, (_, out, _, _) in enumerate(self.ops):
            alloc(out, step)
        lines = [
            "// Generated offline for the native CUDA runtime.",
            *self.kernels,
            f"constexpr size_t INPUT_ELEMENTS = {math.prod(self.shapes['x'])};",
            f"constexpr size_t OUTPUT_ELEMENTS = {math.prod(self.shapes[output_name])};",
            f"constexpr int OUTPUT_ID = {output_id};",
            f"constexpr int MODEL_INPUT_SHAPE[4] = {vec(self.shapes['x'])};",
            f"constexpr int MODEL_OUTPUT_RANK = {len(self.shapes[output_name])};",
            f"constexpr int MODEL_OUTPUT_SHAPE[4] = {vec(list(self.shapes[output_name]) + [1] * (4 - len(self.shapes[output_name])))};",
            f"constexpr size_t ARENA_ELEMENTS = {high};",
            "template<class ModelRuntime> void build_model(ModelRuntime& r) {",
            f"r.t.resize({len(self.tensors)});",
        ]
        weight_offset = 0
        weight_padding = 0
        with (directory / "weights.f32").open("wb") as f:
            for i, name in enumerate(self.tensors):
                if i not in live:
                    continue
                if name in self.constants:
                    v = np.asarray(self.constants[name], dtype="<f4")
                    # Preserve 128-byte alignment after conversion to FP16.
                    # Scalar constants otherwise misalign subsequent GEMM operands.
                    padding = (-weight_offset) % 64
                    f.write(bytes(padding * 4))
                    weight_offset += padding
                    weight_padding += padding
                    f.write(v.tobytes())
                    lines.append(f"r.t[{i}] = r.weights + {weight_offset};")
                    weight_offset += v.size
                else:
                    lines.append(f"r.t[{i}] = r.arena + {offsets[i]};")
        for _, out, code, name in self.ops:
            lines.append(re.sub(r"r\.(\w+)<", r"r.template \1<", code))
        lines.append("}")
        lines.append(f"constexpr size_t WEIGHT_ELEMENTS = {weight_offset};")
        (directory / "model.inc").write_text("\n".join(lines) + "\n")
        manifest = {
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "input_shape": self.shapes["x"],
            "output_shape": self.shapes[output_name],
            "operations": len(self.ops),
            "activation_arena_bytes": high * 2,
            "optimized": self.optimize,
            "fused_pointwise_kernels": sum("void fused_pointwise_" in k for k in self.kernels),
            "indexed_binary_kernels": sum("void indexed_binary_" in k for k in self.kernels),
            "fused_layernorms": self.layernorm_fusions,
            "gemm_epilogues": dict(self.gemm_epilogues),
            "weight_bytes": weight_offset * 4,
            "weight_alignment_elements": 64,
            "weight_padding_bytes": weight_padding * 4,
            "activation_arena_elements": high,
            "arithmetic": "FP16 weights/activations and pointwise GEMM accumulation; FP32 spatial, attention and reduction accumulation",
            "source_operator_counts": getattr(
                self,
                "source_operator_counts",
                dict(collections.Counter(n.op_type for n in self.model.graph.node)),
            ),
            "folded_parallel_convolutions": getattr(self, "folded_convolutions", 0),
            "operation_outputs": [name for _, _, _, name in self.ops],
            "operation_layouts": [
                "nhwc" if out in getattr(self, "nhwc_tensors", set()) else "native"
                for _, out, _, _ in self.ops
            ],
            "channels_last": hasattr(self, "nhwc_tensors"),
            "checkpoints": [
                name
                for _, _, _, name in self.ops
                if name in {n.output[0] for n in self.model.graph.node}
            ],
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in manifest.items()
                    if k not in ("checkpoints", "operation_outputs")
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--width", type=int, default=80)
    p.add_argument("--height", type=int, default=48)
    p.add_argument("--kind", choices=("recognition", "detection"), required=True)
    args = p.parse_args()
    if min(args.batch, args.height, args.width) <= 0:
        p.error("batch, height and width must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    model = onnx.load(args.model)
    source_counts = dict(collections.Counter(n.op_type for n in model.graph.node))
    folded = fold_parallel_convolutions(model) if args.kind == "detection" else 0
    if args.kind == "detection":
        print(f"Folded {folded} parallel convolutions")
    e = Exporter(
        model, args.batch, args.width, True, True, height=args.height
    )
    e.source_operator_counts = source_counts
    e.folded_convolutions = folded
    e.lower()
    e.channels_last()
    e.eliminate_inverse_transposes()
    e.fuse_layernorm()
    e.fuse_pointwise()
    e.fuse_gemm_epilogues(True, True)
    e.specialize_broadcasts()
    e.write(args.out, args.model)
