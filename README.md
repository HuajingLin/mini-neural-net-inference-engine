
# A minimalist neural network inference engine
A minimalist neural network inference engine.    
It helps me understand the core workflow of AI inference and the Data Flow of Neural Networks.    
Written in pure C++17 with no external dependencies. 

[Project WIKI](https://github.com/HuajingLin/mini-neural-net-inference-engine/wiki/mini-neural-network-inference-engine)

## Environment

- Language: CUDA C++ 17; 
- Build System: CMake (>=3.18, requires CUDA language support)

# CMake Build
```
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure

source .venv/bin/activate
python tools/export_mnist.py
deactivate

./build/mnist_infer models/mnist_mlp.bin \
    data/MNIST/raw/t10k-images-idx3-ubyte \
    data/MNIST/raw/t10k-labels-idx1-ubyte

```
# Benchmark 
```
# output example
$ ./build-native/bench_gemm
MxKxN               naive(ijk)             ikj         blocked   (GFLOPS, higher is better)
256x784x128               2.25           23.72           29.58   checksum diff: 1.9e-04
256x128x64                2.99           28.92           27.70   checksum diff: 2.0e-05
256x64x10                 3.57            8.49            7.83   checksum diff: 1.3e-05
1x784x128                 2.26           31.07           28.40   checksum diff: 4.9e-05

# Improvement in inference speed:
$./build/mnist_infer models/mnist_mlp.bin \
    data/MNIST/raw/t10k-images-idx3-ubyte \
    data/MNIST/raw/t10k-labels-idx1-ubyte
accuracy: 9754/10000 = 97.54%
time: 0.135 s (74349 images/s, includes preprocessing)

# before:
# accuracy: 9732/10000 = 97.54%
# time: 0.988 s (10214 images/s, includes preprocessing)

- 0.988 s -> 0.135 Approximately 7.3 times
```

## Project Structure
```
mini_infer/
├── CMakeLists.txt
├── README.md
├── include/mini/
│   ├── tensor.h        # Tensor
│   ├── ops.h           # GEMM
│   ├── layers.h        # layer interface
│   ├── model.h         # Sequential
│   ├── loader.h        # loader
│   └── mnist.h         # read MNIST data
├── src/
│   ├── ops.cpp     
│   ├── layers.cpp 
│   ├── loader.cpp
│   └── mnist.cpp       # read MNIST data
├── tools/
│   ├── gen_gemm_ref.py     # generate matrix
│   ├── gen_ops_ref.py      # generate matrix
│   └── export_mnist.py     # export mnist
├── tests/
│   ├── test_tensor.cpp     # test tensor
│   ├── test_gemm.cpp       # test GEMM
│   ├── test_ops.cpp        # test operators
│   ├── test_model.cpp      # test model
│   ├── test_weights.cpp    # test weights .cpp
│   └── test_gemm_variants.cpp  
├── bench/
│   └── bench_gemm.cpp  
└── apps/ 
    └── mnist_infer.cpp     # inference
```