#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <ctime>

#include "minddata/dataset/engine/datasetops/source/elastic_state.h"
#include <stdml/elastic>
// #include <stdml/bits/data/state2.hpp>

// namespace ml = stdml;
// namespace mpi = ml::collective;
// namespace md = stdml::data;

int get_unix_timestamp() {
  std::time_t t = std::time(0);  // t is an integer type
  return t;
}

namespace kungfu {
ElasticState::ElasticState() {
  // std::string idx_file;
  // int seed = 0;
  // md::state2 ds(idx_file, seed);
}

std::string ElasticState::str() const {
  std::stringstream ss;
  ss << "ElasticState";
  ss << '<' << rank_ << '/' << size_ << '>';
  ss << '@' << progress_;
  return ss.str();
}

bool parse_elastic_state(ElasticState &e) {
  if (auto p = std::getenv("KUNGFU_JOB_START_TIMESTAMP"); p != nullptr) {
    sscanf(p, "%d", &e.job_start_);
  }
  if (auto p = std::getenv("KUNGFU_PROC_START_TIMESTAMP"); p != nullptr) {
    sscanf(p, "%d", &e.proc_start_);
  }
  int t1 = get_unix_timestamp();
  if (true) {
    fprintf(stderr, "%ds since proc start\n", t1 - e.proc_start_);
  }
  if (auto p = std::getenv("KUNGFU_INIT_PROGRESS"); p != nullptr) {
    if (sscanf(p, "%lld", &e.progress_) != 1) {
      return false;
    }
  }
  std::string self;
  std::vector<std::string> peers;
  if (auto p = std::getenv("KUNGFU_SELF_SPEC"); p != nullptr) {
    self = p;
  }
  if (auto p = std::getenv("KUNGFU_INIT_PEERS"); p != nullptr) {
    std::istringstream ss;
    ss.str(p);
    for (std::string line; std::getline(ss, line, ',');) {
      peers.push_back(line);
    }
  }
  e.size_ = peers.size();
  e.rank_ = std::find(peers.begin(), peers.end(), self) - peers.begin();
  if (e.rank_ >= e.size_) {
    return false;
  }
  return true;
}
}  // namespace kungfu
