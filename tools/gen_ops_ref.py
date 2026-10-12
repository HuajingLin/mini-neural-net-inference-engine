import numpy as np

rng = np.random.default_rng(7)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

cases = [(1, 10), (4, 10), (3, 128), (2, 5)]

with open("tests/ops_ref.txt", "w") as f:
    f.write(f"{len(cases)}\n")
    for M, N in cases:
        X = (rng.standard_normal((M, N)) * 5).astype(np.float32)
        b = rng.standard_normal(N).astype(np.float32)
        f.write(f"{M} {N}\n")
        for arr in (X, b, X + b, np.maximum(X, 0), softmax(X)):
            f.write(" ".join(f"{v:.9g}" for v in arr.ravel()) + "\n")
print("written tests/ops_ref.txt")
