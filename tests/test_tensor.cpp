#include <cstdio>
#include <cstdlib>
#include "mini/tensor.h"

#define CHECK(cond) do {\
    if (!(cond)) {\
        std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
        std::exit(1);   \
    } } while (0)

//Verify the correctness of the tensor's element count, 
//memory layout, and element access.
int main() { 
    mini::Tensor t({2, 3}); //2 × 3 2D tensor
    CHECK(t.numel() == 6); 
    t.at(1, 2) = 5.0f; //Write 5.0 in the second row, third column.
    CHECK(t.data()[5] == 5.0f); // Row major order: (1,2) -> 1*3+2 
    std::puts("tensor ok");
}