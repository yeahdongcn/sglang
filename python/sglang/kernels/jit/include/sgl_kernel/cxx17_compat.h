/// Small C++17 compatibility shims for the MUSA mcc frontend.
///
/// mcc 5.2 is based on clang 14 and can crash while parsing libstdc++'s C++20
/// concepts/ranges headers in a device compilation. The JIT wrappers only use
/// a narrow subset of those facilities, so keep the CUDA/ROCm C++20 path intact
/// and provide lightweight equivalents for the MUSA C++17 path.
#pragma once

#if __cplusplus < 202002L

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <new>
#include <type_traits>

#define SGLANG_INTEGRAL typename
#define SGLANG_FLOATING_POINT typename

namespace std {

template <typename T>
using type_identity_t = T;

template <typename T, typename... Args>
constexpr T* construct_at(T* location, Args&&... args) {
  return ::new (static_cast<void*>(location)) T(static_cast<Args&&>(args)...);
}

template <typename T>
class span {
 public:
  using element_type = T;
  using value_type = typename remove_cv<T>::type;
  using iterator = T*;
  using const_iterator = const T*;

  constexpr span() noexcept : m_data(nullptr), m_size(0) {}
  constexpr span(T* data, size_t size) noexcept : m_data(data), m_size(size) {}

  template <typename U, size_t N, typename = enable_if_t<is_convertible_v<U (*)[], T (*)[]>>>
  constexpr span(array<U, N>& data) noexcept : m_data(data.data()), m_size(N) {}

  template <typename U, size_t N, typename = enable_if_t<is_convertible_v<const U (*)[], T (*)[]>>>
  constexpr span(const array<U, N>& data) noexcept : m_data(data.data()), m_size(N) {}

  template <typename U, typename = enable_if_t<is_convertible_v<U (*)[], T (*)[]>>>
  constexpr span(const span<U>& other) noexcept : m_data(other.data()), m_size(other.size()) {}

  template <typename U, typename = enable_if_t<is_convertible_v<const U (*)[], T (*)[]>>>
  constexpr span(initializer_list<U> data) noexcept : m_data(data.begin()), m_size(data.size()) {}

  constexpr T* data() const noexcept { return m_data; }
  constexpr size_t size() const noexcept { return m_size; }
  constexpr bool empty() const noexcept { return m_size == 0; }
  constexpr T& operator[](size_t index) const noexcept { return m_data[index]; }
  constexpr iterator begin() const noexcept { return m_data; }
  constexpr iterator end() const noexcept { return m_data + m_size; }

 private:
  T* m_data;
  size_t m_size;
};

}  // namespace std

#else

#define SGLANG_INTEGRAL std::integral
#define SGLANG_FLOATING_POINT std::floating_point

#endif
