# CMake generated Testfile for 
# Source directory: /home/lin/mini_infer
# Build directory: /home/lin/mini_infer/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(tensor "/home/lin/mini_infer/build/test_tensor")
set_tests_properties(tensor PROPERTIES  _BACKTRACE_TRIPLES "/home/lin/mini_infer/CMakeLists.txt;21;add_test;/home/lin/mini_infer/CMakeLists.txt;0;")
add_test(gemm "/home/lin/mini_infer/build/test_gemm" "/home/lin/mini_infer/tests/gemm_ref.txt")
set_tests_properties(gemm PROPERTIES  _BACKTRACE_TRIPLES "/home/lin/mini_infer/CMakeLists.txt;25;add_test;/home/lin/mini_infer/CMakeLists.txt;0;")
add_test(ops "/home/lin/mini_infer/build/test_ops" "/home/lin/mini_infer/tests/ops_ref.txt")
set_tests_properties(ops PROPERTIES  _BACKTRACE_TRIPLES "/home/lin/mini_infer/CMakeLists.txt;29;add_test;/home/lin/mini_infer/CMakeLists.txt;0;")
