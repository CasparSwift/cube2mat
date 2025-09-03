#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// --------- helper routines ---------
// Compute Gini coefficient for non-negative values in `x`.
static inline float gini_coeff(std::vector<float>& x) {
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

// Compute 1 - normalized entropy for non-negative values in `x`.
static inline float entropy_concentration(const std::vector<float>& x) {
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

// 计算 VWAP: sum(close*volume) / sum(volume)
// 输入: cubes_map["close"], cubes_map["volume"] (float32, shape=(d0,d1,d2))
// 输出: result (float32, shape=(d0,d1))
void cube2mat_vwap(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                   const py::dict& cubes_map) {
    if (!cubes_map.contains("close") || !cubes_map.contains("volume")) {
        throw std::runtime_error("cubes_map must contain 'close' and 'volume'");
    }

    auto close_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["close"]);
    auto volume_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["volume"]);

    auto cbuf = close_arr.request();
    auto vbuf = volume_arr.request();
    if (cbuf.ndim != 3 || vbuf.ndim != 3) {
        throw std::runtime_error("close and volume must be 3D arrays of shape (d0,d1,d2)");
    }
    if (cbuf.shape[0] != vbuf.shape[0] || cbuf.shape[1] != vbuf.shape[1] || cbuf.shape[2] != vbuf.shape[2]) {
        throw std::runtime_error("close and volume must have the same shape");
    }

    const ssize_t d0 = cbuf.shape[0];
    const ssize_t d1 = cbuf.shape[1];
    const ssize_t d2 = cbuf.shape[2];

    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match close/volume leading dims");
    }

    const float* close_ptr  = static_cast<float*>(cbuf.ptr);
    const float* volume_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr          = static_cast<float*>(rbuf.ptr);

    // 索引: idx(i,j,t) = i*(d1*d2) + j*d2 + t
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;

            float num = 0.0f;  // sum(close*volume)
            float den = 0.0f;  // sum(volume)

            for (ssize_t t = 0; t < d2; ++t) {
                float v = volume_ptr[base + t];
                float c = close_ptr [base + t];
                if (std::isnan(v) || std::isnan(c) || !(v > 0.0f)) continue;
                num += c * v;
                den += v;
            }

            res_ptr[i * d1 + j] = (den > 0.0f)
                ? (num / den)
                : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

// ---- gini_absret ---------------------------------------------------------
void cube2mat_gini_absret(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                          const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match last_price leading dims");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> vals;
            vals.reserve(d2 > 0 ? d2 - 1 : 0);
            float prev = price_ptr[base];
            bool prev_valid = std::isfinite(prev) && prev > 0.0f;
            for (ssize_t t = 1; t < d2; ++t) {
                float curr = price_ptr[base + t];
                bool curr_valid = std::isfinite(curr) && curr > 0.0f;
                if (prev_valid && curr_valid) {
                    float r = std::log(curr / prev);
                    float v = std::fabs(r);
                    if (std::isfinite(v) && v >= 0.0f) vals.push_back(v);
                }
                prev = curr;
                prev_valid = curr_valid;
            }
            res_ptr[i * d1 + j] = gini_coeff(vals);
        }
    }
}

// ---- n_entropy_concentration ---------------------------------------------
void cube2mat_n_entropy_concentration(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                      const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'interval_volume'");
    }
    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf = vol_arr.request();
    if (vbuf.ndim != 3) {
        throw std::runtime_error("interval_volume must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = vbuf.shape[0];
    const ssize_t d1 = vbuf.shape[1];
    const ssize_t d2 = vbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match volume leading dims");
    }
    const float* vol_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> vals;
            vals.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (std::isfinite(v) && v > 0.0f) vals.push_back(v);
            }
            res_ptr[i * d1 + j] = entropy_concentration(vals);
        }
    }
}

// ---- n_gini --------------------------------------------------------------
void cube2mat_n_gini(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                     const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'interval_volume'");
    }
    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf = vol_arr.request();
    if (vbuf.ndim != 3) {
        throw std::runtime_error("interval_volume must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = vbuf.shape[0];
    const ssize_t d1 = vbuf.shape[1];
    const ssize_t d2 = vbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match volume leading dims");
    }
    const float* vol_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> vals;
            vals.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (std::isfinite(v) && v >= 0.0f) vals.push_back(v);
            }
            res_ptr[i * d1 + j] = gini_coeff(vals);
        }
    }
}

// ---- ret_skew ------------------------------------------------------------
void cube2mat_ret_skew(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                       const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match last_price leading dims");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> r;
            r.reserve(d2 > 0 ? d2 - 1 : 0);
            float prev = price_ptr[base];
            bool prev_valid = std::isfinite(prev) && prev > 0.0f;
            for (ssize_t t = 1; t < d2; ++t) {
                float curr = price_ptr[base + t];
                bool curr_valid = std::isfinite(curr) && curr > 0.0f;
                if (prev_valid && curr_valid) {
                    double rr = std::log(static_cast<double>(curr) / prev);
                    if (std::isfinite(rr)) r.push_back(rr);
                }
                prev = curr;
                prev_valid = curr_valid;
            }
            const ssize_t n = static_cast<ssize_t>(r.size());
            if (n < 3) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double mean = 0.0;
            for (double v : r) mean += v;
            mean /= static_cast<double>(n);
            double m2 = 0.0, m3 = 0.0;
            for (double v : r) {
                double d = v - mean;
                m2 += d * d;
                m3 += d * d * d;
            }
            double s2 = m2 / static_cast<double>(n - 1);
            if (!(s2 > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double s = std::sqrt(s2);
            double g1 = (static_cast<double>(n) * m3) /
                        ((static_cast<double>(n - 1) * (n - 2)) * s * s * s);
            res_ptr[i * d1 + j] = std::isfinite(g1) ? static_cast<float>(g1)
                                                    : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

// ---- rv_entropy_concentration --------------------------------------------
void cube2mat_rv_entropy_concentration(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                       const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match last_price leading dims");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> sqr;
            sqr.reserve(d2 > 0 ? d2 - 1 : 0);
            float prev = price_ptr[base];
            bool prev_valid = std::isfinite(prev) && prev > 0.0f;
            for (ssize_t t = 1; t < d2; ++t) {
                float curr = price_ptr[base + t];
                bool curr_valid = std::isfinite(curr) && curr > 0.0f;
                if (prev_valid && curr_valid) {
                    double r = std::log(static_cast<double>(curr) / prev);
                    float v = static_cast<float>(r * r);
                    if (std::isfinite(v) && v >= 0.0f) sqr.push_back(v);
                }
                prev = curr;
                prev_valid = curr_valid;
            }
            res_ptr[i * d1 + j] = entropy_concentration(sqr);
        }
    }
}

// ---- rv_gini_concentration -----------------------------------------------
void cube2mat_rv_gini_concentration(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                    const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match last_price leading dims");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> sqr;
            sqr.reserve(d2 > 0 ? d2 - 1 : 0);
            float prev = price_ptr[base];
            bool prev_valid = std::isfinite(prev) && prev > 0.0f;
            for (ssize_t t = 1; t < d2; ++t) {
                float curr = price_ptr[base + t];
                bool curr_valid = std::isfinite(curr) && curr > 0.0f;
                if (prev_valid && curr_valid) {
                    double r = std::log(static_cast<double>(curr) / prev);
                    float v = static_cast<float>(r * r);
                    if (std::isfinite(v) && v >= 0.0f) sqr.push_back(v);
                }
                prev = curr;
                prev_valid = curr_valid;
            }
            res_ptr[i * d1 + j] = gini_coeff(sqr);
        }
    }
}

// ---- trade_size_gini (approximated) -------------------------------------
void cube2mat_trade_size_gini(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                              const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'interval_volume'");
    }
    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf = vol_arr.request();
    if (vbuf.ndim != 3) {
        throw std::runtime_error("interval_volume must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = vbuf.shape[0];
    const ssize_t d1 = vbuf.shape[1];
    const ssize_t d2 = vbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match volume leading dims");
    }
    const float* vol_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> ts;
            ts.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (std::isfinite(v) && v >= 0.0f) ts.push_back(v); // assume n=1 per bar
            }
            res_ptr[i * d1 + j] = gini_coeff(ts);
        }
    }
}

// ---- trend_resid_kurt ----------------------------------------------------
void cube2mat_trend_resid_kurt(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                               const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match last_price leading dims");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> y;
            y.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = price_ptr[base + t];
                if (std::isfinite(v)) y.push_back(v);
            }
            const ssize_t n = static_cast<ssize_t>(y.size());
            if (n < 4) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            // Linear regression y ~ t
            double S_t = 0.0, S_y = 0.0, S_tt = 0.0, S_ty = 0.0;
            for (ssize_t t = 0; t < n; ++t) {
                double tt = static_cast<double>(t) / (n - 1); // 0..1
                double yy = y[t];
                S_t += tt;
                S_y += yy;
                S_tt += tt * tt;
                S_ty += tt * yy;
            }
            double denom = static_cast<double>(n) * S_tt - S_t * S_t;
            if (denom == 0.0) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double beta1 = (static_cast<double>(n) * S_ty - S_t * S_y) / denom;
            double beta0 = (S_y - beta1 * S_t) / static_cast<double>(n);
            std::vector<double> e(n);
            for (ssize_t t = 0; t < n; ++t) {
                double tt = static_cast<double>(t) / (n - 1);
                e[t] = y[t] - (beta0 + beta1 * tt);
            }
            double m = 0.0;
            for (double v : e) m += v;
            m /= static_cast<double>(n);
            double c2 = 0.0, c4 = 0.0;
            for (double v : e) {
                double d = v - m;
                double d2 = d * d;
                c2 += d2;
                c4 += d2 * d2;
            }
            c2 /= static_cast<double>(n);
            c4 /= static_cast<double>(n);
            if (!(c2 > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double g2 = c4 / (c2 * c2) - 3.0;
            double adj = ((static_cast<double>(n) - 1) / ((n - 2.0) * (n - 3.0))) *
                         ((n + 1.0) * g2 + 6.0);
            res_ptr[i * d1 + j] = std::isfinite(adj) ? static_cast<float>(adj)
                                                    : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

// ---- trend_resid_skew ----------------------------------------------------
void cube2mat_trend_resid_skew(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                               const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match last_price leading dims");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> y;
            y.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = price_ptr[base + t];
                if (std::isfinite(v)) y.push_back(v);
            }
            const ssize_t n = static_cast<ssize_t>(y.size());
            if (n < 3) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double S_t = 0.0, S_y = 0.0, S_tt = 0.0, S_ty = 0.0;
            for (ssize_t t = 0; t < n; ++t) {
                double tt = static_cast<double>(t) / (n - 1);
                double yy = y[t];
                S_t += tt;
                S_y += yy;
                S_tt += tt * tt;
                S_ty += tt * yy;
            }
            double denom = static_cast<double>(n) * S_tt - S_t * S_t;
            if (denom == 0.0) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double beta1 = (static_cast<double>(n) * S_ty - S_t * S_y) / denom;
            double beta0 = (S_y - beta1 * S_t) / static_cast<double>(n);
            std::vector<double> e(n);
            for (ssize_t t = 0; t < n; ++t) {
                double tt = static_cast<double>(t) / (n - 1);
                e[t] = y[t] - (beta0 + beta1 * tt);
            }
            double m = 0.0;
            for (double v : e) m += v;
            m /= static_cast<double>(n);
            double c2 = 0.0, c3 = 0.0;
            for (double v : e) {
                double d = v - m;
                double d2 = d * d;
                c2 += d2;
                c3 += d2 * d;
            }
            c2 /= static_cast<double>(n);
            c3 /= static_cast<double>(n);
            if (!(c2 > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double g1 = c3 / std::pow(c2, 1.5);
            double adj = std::sqrt(static_cast<double>(n) * (n - 1.0)) / (n - 2.0);
            double val = adj * g1;
            res_ptr[i * d1 + j] = std::isfinite(val) ? static_cast<float>(val)
                                                     : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

// ---- volume_entropy_concentration ---------------------------------------
void cube2mat_volume_entropy_concentration(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                           const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'interval_volume'");
    }
    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf = vol_arr.request();
    if (vbuf.ndim != 3) {
        throw std::runtime_error("interval_volume must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = vbuf.shape[0];
    const ssize_t d1 = vbuf.shape[1];
    const ssize_t d2 = vbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match volume leading dims");
    }
    const float* vol_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> vals;
            vals.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (std::isfinite(v) && v > 0.0f) vals.push_back(v);
            }
            res_ptr[i * d1 + j] = entropy_concentration(vals);
        }
    }
}

// ---- volume_gini --------------------------------------------------------
void cube2mat_volume_gini(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                          const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'interval_volume'");
    }
    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf = vol_arr.request();
    if (vbuf.ndim != 3) {
        throw std::runtime_error("interval_volume must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = vbuf.shape[0];
    const ssize_t d1 = vbuf.shape[1];
    const ssize_t d2 = vbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match volume leading dims");
    }
    const float* vol_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> vals;
            vals.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (std::isfinite(v) && v >= 0.0f) vals.push_back(v);
            }
            res_ptr[i * d1 + j] = gini_coeff(vals);
        }
    }
}

// ---- volume_median_to_mean_ratio ----------------------------------------
void cube2mat_volume_median_to_mean_ratio(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                          const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'interval_volume'");
    }
    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf = vol_arr.request();
    if (vbuf.ndim != 3) {
        throw std::runtime_error("interval_volume must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = vbuf.shape[0];
    const ssize_t d1 = vbuf.shape[1];
    const ssize_t d2 = vbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match volume leading dims");
    }
    const float* vol_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> vals;
            vals.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (std::isfinite(v) && v >= 0.0f) vals.push_back(v);
            }
            const ssize_t n = static_cast<ssize_t>(vals.size());
            if (n == 0) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double mean = 0.0;
            for (float v : vals) mean += v;
            mean /= static_cast<double>(n);
            if (!(mean > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::sort(vals.begin(), vals.end());
            double median;
            if (n % 2 == 1) {
                median = vals[n / 2];
            } else {
                median = 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
            }
            double ratio = median / mean;
            res_ptr[i * d1 + j] = std::isfinite(ratio) ? static_cast<float>(ratio)
                                                       : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

// ---- volume_skew --------------------------------------------------------
void cube2mat_volume_skew(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                          const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'interval_volume'");
    }
    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf = vol_arr.request();
    if (vbuf.ndim != 3) {
        throw std::runtime_error("interval_volume must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = vbuf.shape[0];
    const ssize_t d1 = vbuf.shape[1];
    const ssize_t d2 = vbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match volume leading dims");
    }
    const float* vol_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> vals;
            vals.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (std::isfinite(v)) vals.push_back(v);
            }
            const ssize_t n = static_cast<ssize_t>(vals.size());
            if (n < 3) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double mean = 0.0;
            for (double v : vals) mean += v;
            mean /= static_cast<double>(n);
            double m2 = 0.0, m3 = 0.0;
            for (double v : vals) {
                double d = v - mean;
                m2 += d * d;
                m3 += d * d * d;
            }
            double s2 = m2 / static_cast<double>(n - 1);
            if (!(s2 > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double s = std::sqrt(s2);
            double m3_mean = m3 / static_cast<double>(n);
            double g1 = m3_mean / (s * s * s);
            res_ptr[i * d1 + j] = std::isfinite(g1) ? static_cast<float>(g1)
                                                    : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

PYBIND11_MODULE(machine_alpha, m) {
    m.doc() = "collection of intraday features";
    m.def("cube2mat_vwap", &cube2mat_vwap,
          py::arg("result"), py::arg("cubes_map"),
          "Compute VWAP over all timesteps for each (i,j).");
    m.def("cube2mat_gini_absret", &cube2mat_gini_absret,
          py::arg("result"), py::arg("cubes_map"),
          "Gini concentration of absolute log returns.");
    m.def("cube2mat_n_entropy_concentration", &cube2mat_n_entropy_concentration,
          py::arg("result"), py::arg("cubes_map"),
          "1 - normalized entropy of trade count distribution.");
    m.def("cube2mat_n_gini", &cube2mat_n_gini,
          py::arg("result"), py::arg("cubes_map"),
          "Gini index of trade count distribution.");
    m.def("cube2mat_ret_skew", &cube2mat_ret_skew,
          py::arg("result"), py::arg("cubes_map"),
          "Sample-adjusted skewness of intraday log returns.");
    m.def("cube2mat_rv_entropy_concentration", &cube2mat_rv_entropy_concentration,
          py::arg("result"), py::arg("cubes_map"),
          "1 - normalized entropy of realized variance contributions.");
    m.def("cube2mat_rv_gini_concentration", &cube2mat_rv_gini_concentration,
          py::arg("result"), py::arg("cubes_map"),
          "Gini concentration of realized variance contributions.");
    m.def("cube2mat_trade_size_gini", &cube2mat_trade_size_gini,
          py::arg("result"), py::arg("cubes_map"),
          "Gini of per-bar average trade size (approximation).");
    m.def("cube2mat_trend_resid_kurt", &cube2mat_trend_resid_kurt,
          py::arg("result"), py::arg("cubes_map"),
          "Adjusted excess kurtosis of trend residuals.");
    m.def("cube2mat_trend_resid_skew", &cube2mat_trend_resid_skew,
          py::arg("result"), py::arg("cubes_map"),
          "Adjusted skewness of trend residuals.");
    m.def("cube2mat_volume_entropy_concentration", &cube2mat_volume_entropy_concentration,
          py::arg("result"), py::arg("cubes_map"),
          "1 - normalized entropy of volume distribution.");
    m.def("cube2mat_volume_gini", &cube2mat_volume_gini,
          py::arg("result"), py::arg("cubes_map"),
          "Gini index of per-bar volume distribution.");
    m.def("cube2mat_volume_median_to_mean_ratio", &cube2mat_volume_median_to_mean_ratio,
          py::arg("result"), py::arg("cubes_map"),
          "Median-to-mean ratio of volume.");
    m.def("cube2mat_volume_skew", &cube2mat_volume_skew,
          py::arg("result"), py::arg("cubes_map"),
          "Skewness of volume distribution.");
}
