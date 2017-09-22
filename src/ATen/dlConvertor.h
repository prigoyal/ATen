#pragma once

#include "ATen/Tensor.h"
#include "dlpack.h"

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor object
// 2) take a dlpack tensor and convert it to the Tensor object

namespace at {

// create the shared pointers typedef
using DLTensorSPtr = std::shared_ptr<DLTensor>;

using ATenTensorSPtr = std::shared_ptr<Tensor>;

class DLConvertor {
  public:
    // constructor for the Tensor types, can be null pointers
    explicit DLConvertor(
      Tensor* atTensor, DLTensor* dlTensor)
      : dlTensor_(dlTensor), atTensor_(atTensor) {}

    // TODO: what is the proper way to destruct this?
    ~DLConvertor() {}

    DLTensorSPtr convertToDLTensor(const ATenTensorSPtr& atTensor);

    ATenTensorSPtr convertToATenTensor(const DLTensorSPtr& dlTensor);

  private:
    // pass the pointers to the dlTensor or the aTensor and get the conversions
    DLTensorSPtr dlTensor_;
    ATenTensorSPtr atTensor_;
};

} //namespace at
