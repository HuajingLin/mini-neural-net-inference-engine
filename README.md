
# A minimalist neural network inference engine
A minimalist neural network inference engine.    
It helps me understand the core workflow of AI inference and bridges the gap between C++ and AI.    
Written in pure C++17 with no external dependencies. 

[Project WIKI](https://github.com/HuajingLin/mini-neural-net-inference-engine/wiki/mini-neural-network-inference-engine)

## Environment

- Language: CUDA C++ 17; 
- Build System: CMake (>=3.18, requires CUDA language support)

# CMake Build
```
python3 tools/gen_gemm_ref.py
python3 tools/gen_ops_ref.py
python tools/export_mnist.py

cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure

```

## output example
```
Test project /home/lin/mini_infer/build
    Start 1: tensor
1/5 Test #1: tensor ...........................   Passed    0.00 sec
    Start 2: gemm
2/5 Test #2: gemm .............................   Passed    0.03 sec
    Start 3: ops
3/5 Test #3: ops ..............................   Passed    0.00 sec
    Start 4: model
4/5 Test #4: model ............................   Passed    0.00 sec
    Start 5: weights
5/5 Test #5: weights ..........................   Passed    0.01 sec

100% tests passed, 0 tests failed out of 5

Total Test time (real) =   0.05 sec

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
│   └── loader.h        # loader
├── src/
│   ├── ops.cpp     
│   ├── layers.cpp
│   └── loader.cpp
├── tools/
│   ├── gen_gemm_ref.py     # generate matrix
│   ├── gen_ops_ref.py      # generate matrix
│   └── export_mnist.py     # export mnist
├── tests/
│   ├── test_tensor.cpp     # test tensor
│   ├── test_gemm.cpp       # test GEMM
│   ├── test_ops.cpp        # test operators
│   ├── test_model.cpp      # test model
│   └── test_weights.cpp    # test weights
├── bench/
└── apps/
```