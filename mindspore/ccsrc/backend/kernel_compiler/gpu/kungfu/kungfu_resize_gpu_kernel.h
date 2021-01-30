#pragma once
#include <map>
#include <string>
#include <vector>

#include "backend/kernel_compiler/cpu/kungfu_common.h"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
class KungFuResizeGpuKernel : public GpuKernel {
 public:
  KungFuResizeGpuKernel()
      : nccl_controller_(nullptr),
        nccl_scheduler_(nullptr),
        input_count_(0),
        output_count_(0),
        input_size_(0),
        output_size_(0),
        workspace_size_(0) {}

  ~KungFuResizeGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }

  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }

  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  bool Init(const CNodePtr &kernel_node) override {
    LOG_InitKernel("KungFuResizeGpuKernel");

    InitResource();
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but requires 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 2) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but requires needs 2 output.";
      return false;
    }

    auto inputA_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto outputC_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);

    InferInAndOutDesc(inputA_shape, outputC_shape);

    InitSizeLists();

    return true;
  }

 protected:
  void InitResource() override {
    const auto nccl_scope_ = KungFu_NCCL_GLOBAL;
    nccl_scheduler_ = _kungfu_nccl_helper->EnsureScheduler(nccl_scope_);
    nccl_controller_ = _kungfu_nccl_helper->EnsureController(nccl_scope_);
  }

  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    output_size_list_.push_back(output_size_);
    return;
  }

 private:
  void DestroyResource() noexcept {}

  void InferInAndOutDesc(const std::vector<size_t> &input_shape, const std::vector<size_t> &output_shape) {
    input_count_ = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
    output_count_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());
    input_size_ = input_count_ * sizeof(uint32_t);
    output_size_ = output_count_ * sizeof(bool);
  }

  kungfu::NCCLController *nccl_controller_;
  kungfu::NCCLScheduler *nccl_scheduler_;

  size_t input_count_;
  size_t output_count_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore
