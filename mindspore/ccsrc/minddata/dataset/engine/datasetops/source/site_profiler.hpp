#pragma once
#include <chrono>
#include <cinttypes>

namespace stdml::utility {

class site_context {
  using C = std::chrono::high_resolution_clock;
  using T = std::chrono::time_point<C>;
  using D = std::chrono::duration<double>;

  const std::string filename_;
  const int lino_;
  const std::string name_;

  int64_t count_;
  D duration_;

  static std::string rel_path(std::string filepath) {
    static const char *prefix = "/home/lg/code/repos/github.com/lgarithm/mindspore";
    filepath.erase(filepath.begin(), filepath.begin() + strlen(prefix));
    return '.' + filepath;
  }

 public:
  site_context(std::string filename, int lino, std::string name)
      : filename_(rel_path(std::move(filename))), lino_(lino), name_(std::move(name)), count_(0), duration_(0) {}

  ~site_context() {
    fprintf(stderr, "%%%% site called %8" PRId64 " times, mean: %6.3fms, total: %8.3fs, %s:%d : %s\n", count_,
            duration_.count() * 1e3 / count_, duration_.count(), filename_.c_str(), lino_, name_.c_str());
  }

  void in() {
    // fprintf(stderr, "entering %s\n", name_.c_str());
  }

  void out(D d) {
    ++count_;
    duration_ += d;
  }
};

class site {
  using C = std::chrono::high_resolution_clock;
  using T = std::chrono::time_point<C>;
  using D = std::chrono::duration<double>;

  site_context &ctx_;
  const T t0_;

 public:
  site(site_context &ctx) : ctx_(ctx), t0_(C::now()) { ctx_.in(); }

  ~site() { ctx_.out(C::now() - t0_); }
};
}  // namespace stdml::utility

#define _PROFILE_SCOPE(name, f, l)                        \
  static stdml::utility::site_context ___ctx(f, l, name); \
  stdml::utility::site ___site(___ctx);

#define _PROFILE_STMT(e, f, l)                            \
  {                                                       \
    static stdml::utility::site_context ___ctx(f, l, #e); \
    stdml::utility::site ___site(___ctx);                 \
    e;                                                    \
  }

#define _PROFILE_EXPR(e, f, l)                            \
  [&] {                                                   \
    static stdml::utility::site_context ___ctx(f, l, #e); \
    stdml::utility::site ___site(___ctx);                 \
    return (e);                                           \
  }()

#define PROFILE_SCOPE(name) _PROFILE_SCOPE(name, __FILE__, __LINE__)
#define PROFILE_STMT(e) _PROFILE_STMT(e, __FILE__, __LINE__)
#define PROFILE_EXPR(e) _PROFILE_EXPR(e, __FILE__, __LINE__)
