
# A minimalist neural network inference engine
A minimalist neural network inference engine.    
It helps me understand the core workflow of AI inference and bridges the gap between C++ and AI.    
Written in pure C++17 with no external dependencies.   

## Environment

- Language: CUDA C++ 17; 
- Build System: CMake (>=3.18, requires CUDA language support)

# CMake Build
python tools/gen_gemm_ref.py
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure

## Project Structure
```
mini_infer/
├── CMakeLists.txt
├── include/mini/
│   └── tensor.h        # L1 Tensor
├── src/
├── tools/
├── tests/test_tensor.cpp   # test tensor
├── bench/
└── apps/
```