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
// const std::map<std::string, cudnnReduceTensorOp_t> kReduceTypeMap = {
//   {"ReduceMax", CUDNN_REDUCE_TENSOR_MAX},
//   {"ReduceMean", CUDNN_REDUCE_TENSOR_AVG},
//   {"ReduceSum", CUDNN_REDUCE_TENSOR_ADD},
//   {"ReduceMin", CUDNN_REDUCE_TENSOR_MIN},
// };

template <typename T>
class KungFuAllReduceGpuKernel : public GpuKernel {
 public:
  KungFuAllReduceGpuKernel()
      : nccl_controller_(nullptr),
        nccl_scheduler_(nullptr),
        comm_stream_(nullptr),
        input_count_(0),
        output_count_(0),
        input_size_(0),
        output_size_(0),
        workspace_size_(0) {}

  ~KungFuAllReduceGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }

  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }

  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    // LOG_Kernel_Launch("KungFuAllReduceCPUKernel", inputs, workspace, outputs);
    KUNGFU_PROFILE_SITE(KungFuAllReduceGpuKernel::Launch);

    const T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    cudaStream_t stream = comm_stream_ ? comm_stream_ : reinterpret_cast<cudaStream_t>(stream_ptr);
    // MS_LOG(WARNING) << "using stream " << stream;

    auto w = make_kungfu_workspace(input_addr, output_addr, input_count_);
    const auto op = KungFu_SUM;  // TODO: support more ops
    // TODO: support async
    nccl_scheduler_->Do([=] { nccl_controller_->AllReduce(w, op, stream); });
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    KUNGFU_PROFILE_SITE(KungFuAllReduceGpuKernel::Init);

    LOG_InitKernel("KungFuAllReduceGpuKernel");
    InitResource();
    data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but requires 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but requires needs 1 output.";
      return false;
    }

    auto inputA_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto outputC_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);

    InferInAndOutDesc(inputA_shape, outputC_shape);

    InitSizeLists();

    // [ERROR] KERNEL(19509,python3.7):2020-11-27-05:25:13.791.838
    // [mindspore/ccsrc/backend/kernel_compiler/gpu/gpu_kernel.h:76] GetAttr] The attr(group) of kernel(KungFuAllReduce)
    // not exist
    // group_name_ = GetAttr<std::string>(kernel_node, kAttrGroup); MS_LOG(WARNING) <<
    // AnfAlgo::GetCNodeName(kernel_node) << " for group " << group_name_;

    auto comm_stream_attr = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("stream_id");
    if (comm_stream_attr) {
      comm_stream_ = reinterpret_cast<cudaStream_t>(GetValue<uintptr_t>(comm_stream_attr));
      MS_EXCEPTION_IF_NULL(comm_stream_);
      MS_LOG(WARNING) << "got kernel_node stream_id: " << comm_stream_;
    }

    return true;
  }

 protected:
  void InitResource() override {
    {
      KUNGFU_PROFILE_SITE(KungFuAllReduceGpuKernel::InitResource::init_kungfu_once);
      init_kungfu_once();
    }
    {
      KUNGFU_PROFILE_SITE(KungFuAllReduceGpuKernel::InitResource::init_kungfu_nccl_once);
      init_kungfu_nccl_once();
    }
    const auto nccl_scope_ = KungFu_NCCL_GLOBAL;
    nccl_scheduler_ = _kungfu_nccl_helper->EnsureScheduler(nccl_scope_);
    nccl_controller_ = _kungfu_nccl_helper->EnsureController(nccl_scope_);
    kungfu::Peer *peer = _kungfu_peer.get();
    {
      KUNGFU_PROFILE_SITE(KungFuAllReduceGpuKernel::InitResource::init_nccl_controller);
      nccl_scheduler_->Do([=] { nccl_controller_->InitOnce(peer); });
    }
  }

  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    return;
  }

 private:
  void DestroyResource() noexcept {
    // noop
  }

  void InferInAndOutDesc(const std::vector<size_t> &input_shape, const std::vector<size_t> &output_shape) {
    input_count_ = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
    output_count_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());
    input_size_ = input_count_ * sizeof(T);
    output_size_ = output_count_ * sizeof(T);
  }

  kungfu::NCCLController *nccl_controller_;
  kungfu::NCCLScheduler *nccl_scheduler_;

  cudaStream_t comm_stream_;
  cudnnDataType_t data_type_;
  std::string group_name_;

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
