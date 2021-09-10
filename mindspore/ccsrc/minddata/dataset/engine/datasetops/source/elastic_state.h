#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <stdml/elastic>
// #include <stdml/bits/data/state2.hpp>

namespace kungfu {
class ElasticState {
  int job_start_;
  int proc_start_;

  int64_t progress_;
  int rank_;
  int size_;

  friend bool parse_elastic_state(ElasticState &e);

 public:
  ElasticState();

  std::string str() const;
};

bool parse_elastic_state(ElasticState &e);

void gen_tf_record();
}  // namespace kungfu
