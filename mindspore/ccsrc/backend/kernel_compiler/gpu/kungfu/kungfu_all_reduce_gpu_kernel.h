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
          input_count_(0),
          output_count_(0),
          input_size_(0),
          output_size_(0),
          workspace_size_(0) {
        MS_LOG(WARNING) << __func__ << " Created";
    }

    ~KungFuAllReduceGpuKernel() override {
        DestroyResource();
        MS_LOG(WARNING) << __func__ << " destroyed";
    }

    const std::vector<size_t> &GetInputSizeList() const override {
        MS_LOG(WARNING) << __func__ << " called ";
        return input_size_list_;
    }

    const std::vector<size_t> &GetOutputSizeList() const override {
        MS_LOG(WARNING) << __func__ << " called";
        return output_size_list_;
    }

    const std::vector<size_t> &GetWorkspaceSizeList() const override {
        MS_LOG(WARNING) << __func__ << " called";
        return workspace_size_list_;
    }

    bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
        MS_LOG(WARNING) << __func__ << " called";
        LOG_Kernel_Launch("KungFuAllReduceCPUKernel", inputs, workspace, outputs);

        T *input_addr = GetDeviceAddress<T>(inputs, 0);
        T *output_addr = GetDeviceAddress<T>(outputs, 0);
        // T *workspace_addr = GetDeviceAddress<T>(workspace, 0);

        MS_LOG(WARNING) << "input_addr " << input_addr;
        MS_LOG(WARNING) << "output_addr " << output_addr;

        auto w = make_kungfu_workspace(input_addr, output_addr, input_count_);
        const auto op = KungFu_SUM;  // TODO: support more ops
        // TODO: support async
        nccl_scheduler_->Do([=] {
            auto done = [] {};
            nccl_controller_->AllReduce(w, op, done);
        });
        return true;
    }

    bool Init(const CNodePtr &kernel_node) override {
        MS_LOG(WARNING) << __func__ << " called";
        LOG_InitKernel("KungFuAllReduceGpuKernel");

        InitResource();

        data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
        size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
        MS_LOG(WARNING) << "data_type_ is " << data_type_;
        MS_LOG(WARNING) << "input_num is " << input_num;
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
        return true;
    }

   protected:
    void InitResource() override {
        MS_LOG(WARNING) << __func__ << " called";
        init_kungfu_once();
        init_kungfu_nccl_once();
        const auto nccl_scope_ = KungFu_NCCL_GLOBAL;

        nccl_scheduler_ = _kungfu_nccl_helper->EnsureScheduler(nccl_scope_);
        MS_LOG(ERROR) << "nccl_scheduler_: " << nccl_scheduler_;

        nccl_controller_ = _kungfu_nccl_helper->EnsureController(nccl_scope_);
        MS_LOG(ERROR) << "nccl_controller_: " << nccl_controller_;

        kungfu::Peer *peer = _kungfu_peer.get();
        nccl_scheduler_->Do([=] { nccl_controller_->InitOnce(peer); });
    }

    void InitSizeLists() override {
        input_size_list_.push_back(input_size_);
        output_size_list_.push_back(output_size_);
        return;
    }

   private:
    void DestroyResource() noexcept {
        // noop
        MS_LOG(WARNING) << __func__ << " called";
    }

    void InferInAndOutDesc(const std::vector<size_t> &input_shape, const std::vector<size_t> &output_shape) {
        input_count_ = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
        output_count_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());

        input_size_ = input_count_ * sizeof(T);
        output_size_ = output_count_ * sizeof(T);
    }

    kungfu::NCCLController *nccl_controller_;
    kungfu::NCCLScheduler *nccl_scheduler_;

    cudnnDataType_t data_type_;

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
