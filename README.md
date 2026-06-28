# micrograd — Learning Notes

Following [Andrej Karpathy's micrograd tutorial](https://github.com/karpathy/micrograd), a from-scratch implementation of a scalar-valued autograd engine and neural network.

## Contents

- **`micrograd/engine.py`** — `Value` class: wraps a scalar, builds a computation graph, and supports reverse-mode autograd (backpropagation).
- **`micrograd/nn.py`** — `Neuron`, `Layer`, and `MLP` built on top of `Value`.
- **`demo1.py`** — training demo with a fixed number of epochs.
- **`demo2.py`** — training demo that stops when loss falls below a threshold.

## Supported Operations

`+` `*` `-` `/` `**` `exp` `tanh`

## Quick Start

```bash
uv run main.py
```

Or run a specific demo:

```bash
uv run demo1.py
uv run demo2.py
```
