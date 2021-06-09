/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDRT_RUNTIME_THREADPOOL_H_
#define MINDSPORE_CORE_MINDRT_RUNTIME_THREADPOOL_H_

#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <new>
#include "thread/threadlog.h"
#include "thread/core_affinity.h"

namespace mindspore {
constexpr int kDefaultFrequency = 1;
constexpr float kMaxScale = 1.;

enum ThreadType { kActorThread = 0, kKernelThread = 1 };

// used in scenarios with unequal division of task
// the parameters indicate the start and end coefficients
using Func = int (*)(void *, int, float, float);
using Content = void *;

typedef struct Task {
  Task(Func f, Content c) : func(f), content(c) {}
  Func func;
  Content content;
  std::atomic_int finished{0};
  std::atomic_int status{THREAD_OK};  // return status, RET_OK
} Task;

typedef struct Worker {
  std::thread thread;
  std::atomic_int type{kActorThread};
  std::atomic_bool active{false};
  std::mutex mutex;
  std::condition_variable cond_var;
  Task *task{nullptr};
  int task_id{0};
  float lhs_scale{0.};
  float rhs_scale{kMaxScale};
  int frequency{kDefaultFrequency};
  int spin{0};
} Worker;

class ThreadPool {
 public:
  static ThreadPool *CreateThreadPool(size_t thread_num);
  virtual ~ThreadPool();

  size_t thread_num() const { return thread_num_; }

  int SetCpuAffinity(const std::vector<int> &core_list);
  int SetCpuAffinity(BindMode bind_mode);

  int SetProcessAffinity(BindMode bind_mode) const;

  int ParallelLaunch(const Func &func, Content content, int task_num);

 protected:
  ThreadPool() = default;

  int CreateThreads(size_t thread_num);
  void DestructThreads();

  int InitAffinityInfo();

  virtual void ThreadAsyncRun(Worker *worker);
  void KernelThreadRun(Worker *worker);

  void SyncRunTask(Task *task, int task_num) const;

  void DistributeTask(Task *task, int task_num);
  void CalculateScales(const std::vector<Worker *> &workers, int sum_frequency) const;
  void ActiveWorkers(const std::vector<Worker *> &workers, Task *task, int task_num) const;

  Worker *CurrentWorker() const;

  std::mutex pool_mutex_;

  std::vector<Worker *> workers_;
  std::vector<Worker *> freelist_;
  std::atomic_bool alive_{true};

  size_t inter_thread_num_{0};
  size_t thread_num_{1};

  CoreAffinity *affinity_{nullptr};
};

}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_THREADPOOL_H_