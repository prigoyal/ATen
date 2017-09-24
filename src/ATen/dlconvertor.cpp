#include "ATen/dlconvertor.h"

#include <iostream>
#include <sstream>

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor object
// 2) take a dlpack tensor and convert it to the Tensor object

using namespace std;
namespace at {
namespace dlpack {

// TODO: probably use macros??
DLDataType DLConvertor::getDLDataType(const Type& type) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = type.elementSizeInBytes() * 8;
  switch (type.scalarType()) {
    case ScalarType::Byte:
      dtype.code = DLDataTypeCode::kUInt;
      break;
    case ScalarType::Char:
      dtype.code = DLDataTypeCode::kInt;
      break;
    case ScalarType::Double:
      dtype.code = DLDataTypeCode::kFloat;
      break;
    case ScalarType::Float:
      dtype.code = DLDataTypeCode::kFloat;
      break;
    case ScalarType::Int:
      dtype.code = DLDataTypeCode::kInt;
      break;
    case ScalarType::Long:
      dtype.code = DLDataTypeCode::kInt;
      break;
    case ScalarType::Short:
      dtype.code = DLDataTypeCode::kInt;
      break;
    case ScalarType::Half:
      dtype.code = DLDataTypeCode::kFloat;
      break;
    case ScalarType::NumOptions:
      throw std::logic_error("NumOptions is not a valid ScalarType");
  }
  return dtype;
}


DLContext DLConvertor::getDLContext(
    const Type& type, const int64_t& device_id) {
  DLContext ctx;
  ctx.device_id = device_id;
  if (type.isCuda()) {
    ctx.device_type = DLDeviceType::kGPU;
  } else {
    ctx.device_type = DLDeviceType::kCPU;
  }
  return ctx;
}


int64_t* DLConvertor::getDLInt64Array(const IntList& arr) {
  size_t arrLen = arr.size();
  auto out = new int64_t[arrLen];
  for (size_t i = 0; i < arrLen; i++) {
    out[i] = arr[i];
  }
  return out;
}


// This function returns a shared_ptr to DLpack tensor constructed out ATen tensor
DLTensorSPtr DLConvertor::convertToDLTensor(const Tensor& atTensor) {
  DLTensorSPtr dlTensor(new DLTensor);
  dlTensor->data = atTensor.data_ptr();
  // TODO: get_device() throws error
  // int64_t device_id = atTensor.get_device();
  int64_t device_id = 0;

  dlTensor->ctx = getDLContext(atTensor.type(), device_id);
  dlTensor->ndim = atTensor.dim();
  dlTensor->dtype = getDLDataType(atTensor.type());
  dlTensor->shape = getDLInt64Array(atTensor.sizes());
  dlTensor->strides = getDLInt64Array(atTensor.strides());
  // TODO: what is the correct offset?
  dlTensor->byte_offset = 0;
  return dlTensor;
}


Backend DLConvertor::getATenBackend(const DLContext& ctx) {
  Backend backend;
  switch (ctx.device_type) {
    case DLDeviceType::kCPU:
      backend = Backend::CPU;
      break;
    case DLDeviceType::kGPU:
      backend = Backend::CUDA;
      break;
    default:
      throw std::logic_error("Unsupported device_type: " + std::to_string(ctx.device_type));
  }
  return backend;
}


// TODO: use macros?
ScalarType DLConvertor::getATenScalarType(const DLDataType& dtype) {
  ScalarType stype;
  if (dtype.lanes != 1) throw std::logic_error("ATen does not support lanes != 1");
  switch (dtype.code) {
    case DLDataTypeCode::kUInt:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Byte;
          break;
        default:
          throw std::logic_error("Unsupported kUInt bits " + std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kInt:
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Char;
          break;
        case 16:
          stype = ScalarType::Short;
          break;
        case 32:
          stype = ScalarType::Int;
          break;
        case 64:
          stype = ScalarType::Long;
          break;
        default:
          throw std::logic_error("Unsupported kInt bits " + std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kFloat:
      switch (dtype.bits) {
        case 16:
          stype = ScalarType::Half;
          break;
        case 32:
          stype = ScalarType::Float;
          break;
        case 64:
          stype = ScalarType::Double;
          break;
        default:
          throw std::logic_error("Unsupported kFloat bits " + std::to_string(dtype.bits));
      }
      break;
    default:
      throw std::logic_error("Unsupported code " + std::to_string(dtype.code));
  }
  return stype;
}


Tensor DLConvertor::convertToATenTensor(const DLTensorSPtr& dlTensor) {
  Backend backend = getATenBackend(dlTensor->ctx);
  ScalarType stype = getATenScalarType(dlTensor->dtype);
  return getType(backend, stype).tensorFromBlob(
      dlTensor->data,
      IntList(dlTensor->shape, dlTensor->ndim),
      IntList(dlTensor->strides, dlTensor->ndim));
}

} // namespace dlpack
} //namespace at
