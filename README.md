
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
python3 tools/gen_gemm_ref.py
python3 tools/gen_ops_ref.py
python tools/export_mnist.py

cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure

```

## output example
```






                        ++**
                    --##%%%%
                    %%%%%%%%++
                  ++%%%%%%%%%%%%==
                ..%%%%##**%%%%%%%%
                ::%%**    --##%%@@##
                --%%::      **%%%%**
                **%%++      %%%%%%==
                --%%##..--++%%%%%%%%..
                  %%%%%%%%%%**..##%%++
                  ::%%@@%%++    ==%%##..
                    ::++++        **%%++
                                  --%%##
                                    ++%%==
                                    ..%%%%..
                                      ==%%**
                                        ##%%..
                                        ..%%++
                                          **%%..
                                            ##::



image #7   label: 9   prediction: 9   (correct)
 0: 0.0000
 1: 0.0000
 2: 0.0001
 3: 0.0031
 4: 0.0001
 5: 0.0000
 6: 0.0000
 7: 0.0029
 8: 0.0000
 9: 0.9939

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
│   └── test_weights.cpp    # test weights
├── bench/
└── apps/ 
    └── mnist_infer.cpp     # inference
```