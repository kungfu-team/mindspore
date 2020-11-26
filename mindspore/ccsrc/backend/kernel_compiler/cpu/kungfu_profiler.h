#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace kungfu {
class site_profiler {
  using C = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<long, std::nano>;
  using Instant = std::chrono::time_point<C>;

  static std::string show_duration(const Duration &d) {
    std::stringstream ss;
    ss << std::setprecision(4) << std::setw(6);
    if (d < std::chrono::microseconds(1)) {
      ss << d.count() << "ns";
    } else if (d < std::chrono::milliseconds(1)) {
      using D = std::chrono::duration<float, std::micro>;
      auto d1 = std::chrono::duration_cast<D>(d);
      ss << d1.count() << "us";
    } else if (d < std::chrono::seconds(1)) {
      using D = std::chrono::duration<float, std::milli>;
      auto d1 = std::chrono::duration_cast<D>(d);
      ss << d1.count() << "ms";
    } else {
      using D = std::chrono::duration<float>;
      auto d1 = std::chrono::duration_cast<D>(d);
      ss << d1.count() << "s";
    }
    return ss.str();
  }

  static std::string show(int n, int w = 6) {
    std::stringstream ss;
    ss << std::setw(w);
    ss << n;
    return ss.str();
  }

  const int report_period_ = 0;
  const std::string name_;

  Duration min_;
  Duration max_;
  Duration total_;
  size_t n_;

 public:
  class scope {
    site_profiler &ctx_;
    const Instant t0_;

   public:
    scope(site_profiler &ctx) : ctx_(ctx), t0_(C::now()) {}

    ~scope() {
      Duration d = C::now() - t0_;
      ctx_.end(d);
    }
  };

  site_profiler(int line) : site_profiler("LINE:" + std::to_string(line)) {}

  site_profiler(std::string name) : name_(std::move(name)) { reset(); }

  void reset() {
    min_ = std::chrono::hours(1);
    max_ = std::chrono::hours(0);
    total_ = std::chrono::hours(0);
    n_ = 0;
  }

  void report() {
    const auto mean_ = total_ / n_;
    std::cout << "called " << show(n_) << " times "   //
              << " mean: " << show_duration(mean_)    //
              << " total: " << show_duration(total_)  //
              << " min: " << show_duration(min_)      //
              << " max: " << show_duration(max_)      //
              << " call site: " << name_              //
              << std::endl;
  }

  ~site_profiler() {
    if (n_ > 0) {
      report();
    }
  }

  void begin() {}

  void end(Duration d) {
    if (d < min_) {
      min_ = d;
    }
    if (d > max_) {
      max_ = d;
    }
    total_ += d;
    ++n_;

    if (report_period_ > 0 && n_ % report_period_ == 0) {
      report();
      reset();
    }
  }
};
}  // namespace kungfu

#define KUNGFU_PROFILE_SITE(e)          \
  static kungfu::site_profiler ctx(#e); \
  kungfu::site_profiler::scope __profile_site(ctx);
