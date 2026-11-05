#pragma once
// Minimal self-contained test helpers (no external test framework needed).

#include <cmath>
#include <cstdio>

namespace testing_detail {
inline int& failures() {
    static int n = 0;
    return n;
}
inline int& checks() {
    static int n = 0;
    return n;
}
}  // namespace testing_detail

#define CHECK(cond)                                                                  \
    do {                                                                             \
        ++testing_detail::checks();                                                  \
        if (!(cond)) {                                                               \
            ++testing_detail::failures();                                            \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
        }                                                                            \
    } while (0)

// |a - b| <= tol
#define CHECK_NEAR(a, b, tol)                                                        \
    do {                                                                             \
        ++testing_detail::checks();                                                  \
        const double va_ = (a), vb_ = (b), vt_ = (tol);                              \
        if (!(std::abs(va_ - vb_) <= vt_)) {                                         \
            ++testing_detail::failures();                                            \
            std::printf("FAIL %s:%d: %s = %.12g, %s = %.12g, |diff| = %.3g > %.3g\n", \
                        __FILE__, __LINE__, #a, va_, #b, vb_, std::abs(va_ - vb_), vt_); \
        }                                                                            \
    } while (0)

inline int test_summary(const char* name) {
    std::printf("%s: %d checks, %d failures\n", name, testing_detail::checks(), testing_detail::failures());
    return testing_detail::failures() == 0 ? 0 : 1;
}
