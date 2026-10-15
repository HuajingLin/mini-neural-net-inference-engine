import struct

import numpy as np
from torchvision import datasets, transforms


def load_tensors(path):
    out = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"MINI"
        _version, count = struct.unpack("<II", f.read(8))
        for _ in range(count):
            (n,) = struct.unpack("<I", f.read(4))
            name = f.read(n).decode()
            (ndim,) = struct.unpack("<I", f.read(4))
            shape = struct.unpack(f"<{ndim}I", f.read(4 * ndim))
            numel = int(np.prod(shape))
            out[name] = np.frombuffer(f.read(4 * numel), dtype="<f4").reshape(shape)
    return out


def forward(w, x):
    x = np.maximum(x @ w["fc1.weight"] + w["fc1.bias"], 0)
    x = np.maximum(x @ w["fc2.weight"] + w["fc2.bias"], 0)
    return x @ w["fc3.weight"] + w["fc3.bias"]


tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])
test_set = datasets.MNIST("data", train=False, download=False, transform=tf)
labels = test_set.targets.numpy()
n = len(test_set)

# 1) torchvision preprocessing
x_tv = np.stack([test_set[i][0].numpy().reshape(-1) for i in range(n)])

# 2) Manual preprocessing from raw bytes
raw = test_set.data.numpy().reshape(n, -1).astype(np.float32)
x_manual = ((raw / np.float32(255) - np.float32(0.1307)) / np.float32(0.3081)).astype(np.float32)

w = load_tensors("models/mnist_mlp.bin")
for name, x in (("torchvision preprocessing", x_tv), ("manual preprocessing", x_manual)):
    pred = forward(w, x).argmax(1)
    print(f"{name}: {(pred == labels).sum()}/{n}")
print("max |x_tv - x_manual| =", np.abs(x_tv - x_manual).max())