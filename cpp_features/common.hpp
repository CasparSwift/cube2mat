#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

inline float gini_coeff(std::vector<float>& x) {
    const ssize_t n = static_cast<ssize_t>(x.size());
    if (n < 2) return std::numeric_limits<float>::quiet_NaN();
    double sum = 0.0;
    for (float v : x) sum += v;
    if (!(sum > 0.0)) return std::numeric_limits<float>::quiet_NaN();
    std::sort(x.begin(), x.end());
    double g = 0.0;
    for (ssize_t k = 0; k < n; ++k) {
        g += (static_cast<double>(n - k) - 0.5) * x[k];
    }
    g = 1.0 - 2.0 * g / (static_cast<double>(n) * sum);
    if (!std::isfinite(g)) return std::numeric_limits<float>::quiet_NaN();
    if (g < 0.0) g = 0.0; else if (g > 1.0) g = 1.0;
    return static_cast<float>(g);
}

inline float entropy_concentration(const std::vector<float>& x) {
    const ssize_t m = static_cast<ssize_t>(x.size());
    double tot = 0.0;
    for (float v : x) tot += v;
    if (m < 2 || !(tot > 0.0)) return std::numeric_limits<float>::quiet_NaN();
    double H = 0.0;
    for (float v : x) {
        double p = static_cast<double>(v) / tot;
        if (p > 0.0) H -= p * std::log(p);
    }
    double Hmax = std::log(static_cast<double>(m));
    if (!(Hmax > 0.0)) return std::numeric_limits<float>::quiet_NaN();
    double conc = 1.0 - H / Hmax;
    if (!std::isfinite(conc)) return std::numeric_limits<float>::quiet_NaN();
    if (conc < 0.0) conc = 0.0; else if (conc > 1.0) conc = 1.0;
    return static_cast<float>(conc);
}
