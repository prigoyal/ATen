#pragma once

#include "dlConvertor.h"

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor object
// 2) take a dlpack tensor and convert it to the Tensor object

namespace at { namespace dlpack {

DLDataType getDLDataType(const ATenTensorSPtr& atTensor) {
  DLDataType dtype;
  dtype.lanes = 1;
  switch (atTensor->scalarType()) {
    case ScalarType::Byte:
      dtype.code = DLDataTypeCode::kUInt;
      dtype.bits = 8;
      break;
    case ScalarType::Char:
      dtype.code = DLDataTypeCode::kInt;
      dtype.bits = 8;
      break;
    case ScalarType::Double:
      dtype.code = DLDataTypeCode::kFloat;
      dtype.bits = 64;
      break;
    case ScalarType::Float:
      dtype.code = DLDataTypeCode::kFloat;
      dtype.bits = 32;
      break;
    case ScalarType::Int:
      dtype.code = DLDataTypeCode::kInt;
      dtype.bits = 32;
      break;
    case ScalarType::Long:
      dtype.code = DLDataTypeCode::kInt;
      dtype.bits = 64;
      break;
    case ScalarType::Short:
      dtype.code = DLDataTypeCode::kInt;
      dtype.bits = 16;
      break;
    case ScalarType::Half:
      dtype.code = DLDataTypeCode::kFloat;
      dtype.bits = 16;
      break;
    case ScalarType::NumOptions:
      throw std::logic_error("NumOptions is not a valid ScalarType");
  }
  return dtype;
}

DLContext getDLContext(const ATenTensorSPtr& atTensor) {
  DLContext ctx;
  if (atTensor.type().isCuda()) {
    ctx.device_type = DLDeviceType::kGPU;
  } else {
    ctx.device_type = DLDeviceType::kCPU;
  }
  ctx.device_id = atTensor->get_device();
  return ctx;
}


int64_t* getDLInt64Array(const IntList& arr) {
  auto arrLen = arr.size();
  auto out = new int64_t[arrLen];
  for (size_t i = 0; i < arrLen; i++) {
    out[i] = arrLen[i];
  }
  return out;
}


ScalarType getATenScalarType(const DLDataType& dtype) {
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

Backend getATenBackend(const DLContext& ctx) {
  Backend backend;
  switch (ctx.device_type) {
    case DLDeviceType::kCPU:
      backend = Backend::CPU;
    case DLDeviceType::kGPU:
      backend = Backend::CUDA;
    default:
      throw std::logic_error("Unsupported device_type" + std::to_string(ctx.device_type));
  }
  return backend;
}

// This function takes shared pointer to ATen Tensor and returns a shared_ptr
// to DLpack tensor constructed out of it
DLTensorSPtr convertToDLTensor(Tensor& atTensor) {
  DLTensorSPtr dlTensor(new DLTensor);
  dlTensor->data = atTensor.data_ptr();
  dlTensor->ctx = getDLContext(atTensor.type());
  dlTensor->ndim = atTensor.dim();
  dlTensor->dtype = getDLDataType(atTensor.type());
  dlTensor->shape = getDLInt64Array(atTensor.sizes());
  dlTensor->strides = getDLInt64Array(atTensor.strides());
  dlTensor->byte_offset = 0;  // TODO: what is the correct offset?
  return dlTensor;
}


// This function takes shared pointer to ATen Tensor and returns a shared_ptr
// to DLpack tensor constructed out of it
ATenTensorSPtr convertToATenTensor(Tensor& dlTensor) {
  ATenTensorSPtr atTensor(new Tensor);
  Backend backend = getATenBackend(dlTensor.ctx);
  ScalarType stype = getATenScalarType(dlTensor.dtype);
  Type* type = getType(backend, stype);
  return atTensor;
}

}} //namespace at::dlpack
