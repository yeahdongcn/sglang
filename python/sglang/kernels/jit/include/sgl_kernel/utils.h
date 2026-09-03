/// \file utils.h
/// \brief Host-side C++ utilities used by JIT kernel wrappers.

#pragma once

// ref: https://forums.developer.nvidia.com/t/c-20s-source-location-compilation-error-when-using-nvcc-12-1/258026/3
#ifdef __CUDACC__
#include <cuda.h>
#if CUDA_VERSION <= 12010

#pragma push_macro("__cpp_consteval")
#pragma push_macro("_NODISCARD")
#pragma push_macro("__builtin_LINE")

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbuiltin-macro-redefined"
#define __cpp_consteval 201811L
#pragma clang diagnostic pop

#ifdef _NODISCARD
#undef _NODISCARD
#define _NODISCARD
#endif

#define consteval constexpr

#include "source_location.h"

#undef consteval
#pragma pop_macro("__cpp_consteval")
#pragma pop_macro("_NODISCARD")
#else  // __CUDACC__ && CUDA_VERSION > 12010
#include "source_location.h"
#endif
#else  // no __CUDACC__
#include "source_location.h"
#endif

#include <dlpack/dlpack.h>

#include <sgl_kernel/cxx17_compat.h>

#if __cplusplus >= 202002L
#include <concepts>
#endif
#include <cstddef>
#include <ostream>
#if __cplusplus >= 202002L
#include <ranges>
#endif
#include <sstream>
#include <utility>

namespace sglang {

namespace host {

template <typename>
inline constexpr bool dependent_false_v = false;

/// \brief Source-location wrapper for debug/error messages.
struct DebugInfo : public source_location_t {
  DebugInfo(source_location_t loc = source_location_t::current()) : source_location_t(loc) {}
};

/// \brief Exception type thrown by `RuntimeCheck` and `Panic`.
struct PanicError : public std::runtime_error {
 public:
  explicit PanicError(std::string msg) : runtime_error(msg), m_message(std::move(msg)) {}
  auto root_cause() const -> std::string_view {
    const auto str = std::string_view{m_message};
    const auto pos = str.find(": ");
    return pos == std::string_view::npos ? str : str.substr(pos + 2);
  }

 private:
  std::string m_message;
};

/// \brief Unconditionally abort with a formatted error message.
template <typename... Args>
[[noreturn]]
inline auto panic(DebugInfo location, Args&&... args) -> void {
  std::ostringstream os;
  os << "Failed at " << location.file_name() << ":" << location.line();
  if constexpr (sizeof...(args) > 0) {
    os << ": ";
    (os << ... << std::forward<Args>(args));
  } else {
    os << " in " << location.function_name();
  }
  throw PanicError(std::move(os).str());
}

/**
 * \brief Runtime assertion: panics with a formatted message when `condition`
 *        is false. Extra `args` are streamed to the error message.
 *
 * Example:
 * \code
 *   RuntimeCheck(n > 0, "n must be positive, got ", n);
 * \endcode
 */
template <typename... Args>
struct RuntimeCheck {
  template <typename Cond>
  explicit RuntimeCheck(Cond&& condition, Args&&... args, DebugInfo location = {}) {
    if (condition) return;
    [[unlikely]] host::panic(location, std::forward<Args>(args)...);
  }
  template <typename Cond>
  explicit RuntimeCheck(DebugInfo location, Cond&& condition, Args&&... args) {
    if (condition) return;
    [[unlikely]] host::panic(location, std::forward<Args>(args)...);
  }
};

template <typename... Args>
struct Panic {
  explicit Panic(Args&&... args, DebugInfo location = {}) {
    host::panic(location, std::forward<Args>(args)...);
  }
  explicit Panic(DebugInfo location, Args&&... args) {
    host::panic(location, std::forward<Args>(args)...);
  }
  [[noreturn]] ~Panic() {
    std::terminate();
  }
};

template <typename Cond, typename... Args>
explicit RuntimeCheck(Cond&&, Args&&...) -> RuntimeCheck<Args...>;

template <typename Cond, typename... Args>
explicit RuntimeCheck(DebugInfo, Cond&&, Args&&...) -> RuntimeCheck<Args...>;

template <typename... Args>
explicit Panic(Args&&...) -> Panic<Args...>;

template <typename... Args>
explicit Panic(DebugInfo, Args&&...) -> Panic<Args...>;

namespace pointer {

// we only allow void * pointer arithmetic for safety

template <typename T = char, SGLANG_INTEGRAL... U>
inline auto offset(void* ptr, U... offset) -> void* {
  return static_cast<T*>(ptr) + (... + offset);
}

template <typename T = char, SGLANG_INTEGRAL... U>
inline auto offset(const void* ptr, U... offset) -> const void* {
  return static_cast<const T*>(ptr) + (... + offset);
}

}  // namespace pointer

/// \brief Integer ceiling division: ceil(a / b).
template <SGLANG_INTEGRAL T, SGLANG_INTEGRAL U>
inline constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

/// \brief Returns the byte width of a DLPack data type.
inline auto dtype_bytes(DLDataType dtype) -> std::size_t {
  return static_cast<std::size_t>(dtype.bits / 8);
}

#if __cplusplus >= 202002L
namespace stdr = std::ranges;
namespace stdv = stdr::views;
#else
namespace stdr {
template <typename R>
inline auto empty(const R& range) -> bool {
  return range.begin() == range.end();
}

template <typename R, typename T>
inline auto find(const R& range, const T& value) -> decltype(range.begin()) {
  return std::find(range.begin(), range.end(), value);
}

template <typename R>
inline auto end(const R& range) -> decltype(range.end()) {
  return range.end();
}

template <typename R, typename Predicate>
inline auto any_of(const R& range, Predicate predicate) -> bool {
  return std::any_of(range.begin(), range.end(), predicate);
}

template <typename InputIt, typename Size, typename OutputIt>
inline auto copy_n(InputIt first, Size count, OutputIt result) -> OutputIt {
  return std::copy_n(first, count, result);
}
}  // namespace stdr
#endif

#if __cplusplus < 202002L
template <typename T>
class IntegerRange {
 private:
  class Iterator {
   public:
    explicit Iterator(T value) : m_value(value) {}
    T operator*() const { return m_value; }
    Iterator& operator++() {
      ++m_value;
      return *this;
    }
    bool operator!=(const Iterator& other) const { return m_value != other.m_value; }

   private:
    T m_value;
  };

 public:
  IntegerRange(T start, T end) : m_start(start), m_end(end) {}
  Iterator begin() const { return Iterator(m_start); }
  Iterator end() const { return Iterator(m_end); }

 private:
  T m_start;
  T m_end;
};
#endif

/// \brief Python-style integer range: `irange(n)` -> `[0, n)`.
template <SGLANG_INTEGRAL T>
inline auto irange(T end) {
#if __cplusplus >= 202002L
  return stdv::iota(static_cast<T>(0), end);
#else
  return IntegerRange<T>(static_cast<T>(0), end);
#endif
}

/// \brief Python-style integer range: `irange(start, end)` -> `[start, end)`.
template <SGLANG_INTEGRAL T>
inline auto irange(T start, T end) {
#if __cplusplus >= 202002L
  return stdv::iota(start, end);
#else
  return IntegerRange<T>(start, end);
#endif
}

/** \brief Error class for stream-style error logging. */
struct Error {
  Error(DebugInfo location = {}) {
    m_oss << "Failed at " << location.file_name() << ":" << location.line() << ": ";
  }

  template <typename T>
  Error& operator<<(T&& arg) {
    m_oss << std::forward<T>(arg);
    return *this;
  }

  [[noreturn]]
  ~Error() noexcept(false) {
    throw PanicError(std::move(m_oss).str());
  }

 private:
  std::ostringstream m_oss;
};

/**
 * \brief 0-overhead CHECK macro for host code. This can avoid unnecessary
 * instantiation of error messages when the condition is true.
 *
 * Usage: CHECK_HOST(ptr != nullptr) << "Pointer must not be null";
 */
// The empty-true-branch if/else form keeps a trailing `else` in user code
// bound to the user's `if`, not to the macro's.
#define CHECK_HOST(COND) \
  if (COND) [[likely]] { \
  } else                 \
    host::Error()

}  // namespace host

}  // namespace sglang
