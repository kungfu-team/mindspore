/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstring>
#include <algorithm>
#include <set>
#include<string>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/cast.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/cast.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Cast;

namespace mindspore::kernel {

int CastOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t im_dst_x, im_dst_y;
  if (in_tensors_[0]->GetFormat() == schema::Format::Format_NHWC4) {
    im_dst_x = out_tensors_[0]->Width() * CO4;
    im_dst_y = out_tensors_[0]->Height();
  } else {
    im_dst_y = out_tensors_[0]->Batch() * out_tensors_[0]->Height() * CO4;
    im_dst_x = out_tensors_[0]->Width();
  }
  size_t img_dtype = CL_FLOAT;
  auto enable_fp16_ = ocl_runtime_->GetFp16Enable();
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

void CastOpenCLKernel::GetKernelName(std::string *kernel_name, CastParameter *param) {
  if (param->src_type_ == kNumberTypeFloat32 && param->dst_type_ == kNumberTypeFloat16) {
    kernel_name[0] += "_Fp32ToFp16";
  } else if (param->src_type_ == kNumberTypeFloat16 && param->dst_type_ == kNumberTypeFloat32) {
    kernel_name[0] += "_Fp16ToFp32";
  } else {
    MS_LOG(ERROR) << "unsupported convert format from : " << param->src_type_ << "to  " << param->dst_type_;
  }
}

int CastOpenCLKernel::Init() {
  auto param = reinterpret_cast<CastParameter *>(this->op_parameter_);
  auto in_format = op_format_;
  if (in_format != schema::Format_NHWC4 && in_format != schema::Format_NC4HW4) {
    MS_LOG(ERROR) << "input format(" << in_format << ") "
                  << "format not support!";
    return RET_ERROR;
  }
  in_ori_format_ = in_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(op_format_);
  out_ori_format_ = out_tensors_[0]->GetFormat();
  out_tensors_[0]->SetFormat(op_format_);
  std::string kernel_name = "Cast";
  GetKernelName(&kernel_name, param);
  if (in_format == schema::Format_NC4HW4) {
    kernel_name += "_NC4HW4";
  } else if (in_format == schema::Format_NHWC4) {
    kernel_name += "_NHWC4";
  }
  std::set<std::string> build_options;
  std::string source = cast_source;
  std::string program_name = "cast";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);

  return RET_OK;
}

int CastOpenCLKernel::ReSize() { return RET_OK; }

void CastGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
  const int max_divider = 8;
  const int max_x = 4, max_y = 8;
  int x = std::min(GetMaxDivisorStrategy1(global[0], max_divider), max_x);
  int yz = max_size / x;
  int y = std::min(std::min(GetMaxDivisorStrategy1(global[1], max_divider), yz), max_y);
  int z = std::min(yz / y, static_cast<int>(UP_DIV(global[2], 2)));

  local->clear();
  local->push_back(x);
  local->push_back(y);
  local->push_back(z);
}

int CastOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  auto input_shape = in_tensors_[0]->shape();
  cl_int4 input_shape_ = {input_shape[0], input_shape[1], input_shape[2], UP_DIV(input_shape[3], C4NUM)};

  uint32_t OH = input_shape[1];
  uint32_t OW = input_shape[2];
  uint32_t OC = UP_DIV(input_shape[3], C4NUM);

  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  std::vector<size_t> local = {1, 1, 1};  // init local
  std::vector<size_t> global = {OH, OW, OC};
  CastGetWorkGroup(global, &local, max_global[0]);
  int arg_cn = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_[0]->data_c());   // input tensor
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c());  // out tensor
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_shape_);
  ocl_runtime_->RunKernel(kernel_, global, local, nullptr);

  return RET_OK;
}

kernel::LiteKernel *OpenCLCastKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                            const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                            const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) CastOpenCLKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << " new CastOpenCLKernel failed ";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << " Init kernel failed, name: Cast ";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Cast, OpenCLCastKernelCreator);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Cast, OpenCLCastKernelCreator);
}  // namespace mindspore::kernel
