#include "ATen/ATen.h"
#include "ATen/dlconvertor.h"

#include <iostream>
#include <string.h>
#include <sstream>
#include "test_assert.h"

using namespace at;

static void test() {
  {
    std::cout << "dlconvertor: convert ATen to DLTensor" << std::endl;
    Tensor a = CPU(at::kFloat).rand({3,4});
    std::cout << a.numel() << std::endl;
    dlpack::DLConvertor convertor(a);
    dlpack::DLTensorSPtr dlTensor = convertor.convertToDLTensor(a);
    std::cout << "dlconvertor: convert DLTensor to ATen" << std::endl;
    Tensor b = convertor.convertToATenTensor(dlTensor);
    ASSERT(a.equal(b));
    std::cout << "conversion was fine" << std::endl;
  }

}

int main(int argc, char ** argv)
{
  std::cout << "======================= CPU =====================" << std::endl;
  test();
  return 0;
}
