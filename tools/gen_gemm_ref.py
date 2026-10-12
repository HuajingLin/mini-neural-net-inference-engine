import numpy as np

rng = np.random.default_rng(42)
cases = [(1, 1, 1), (2, 3, 2), (5, 7, 3), (16, 784, 128), (1, 128, 10)]

with open("tests/gemm_ref.txt", "w") as f: 
    f.write(f"{len(cases)}\n") 
    #Extract each tuple in turn and assign the three values ​​to M, K, and N, respectively.
    for M, K, N in cases:
        # Generate matrices A and B.
        A = rng.standard_normal((M, K)).astype(np.float32) 
        B = rng.standard_normal((K, N)).astype(np.float32) 
        # @ is the matrix multiplication operator in Python.
        C = A @ B # float32 result 
        f.write(f"{M} {K} {N}\n") 
        for arr in (A, B, C): 
            f.write(" ".join(f"{x:.9g}" for x in arr.ravel()) + "\n")
print("written tests/gemm_ref.txt")