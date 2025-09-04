#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_trend_resid_skew(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                               const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double sum_t = 0.0, sum_p = 0.0, sum_tt = 0.0, sum_tp = 0.0;
            ssize_t count = 0;
            for (ssize_t t = 0; t < d2; ++t) {
                float p = price_ptr[base + t];
                if (!std::isnan(p)) {
                    sum_t += t;
                    sum_p += p;
                    sum_tt += t * t;
                    sum_tp += t * p;
                    ++count;
                }
            }
            if (count < 3) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double mean_t = sum_t / count;
            double mean_p = sum_p / count;
            double cov_tp = sum_tp / count - mean_t * mean_p;
            double var_t = sum_tt / count - mean_t * mean_t;
            double slope = (var_t > 0.0) ? cov_tp / var_t : 0.0;
            double intercept = mean_p - slope * mean_t;
            std::vector<double> resid; resid.reserve(count);
            for (ssize_t t = 0; t < d2; ++t) {
                float p = price_ptr[base + t];
                if (!std::isnan(p)) {
                    double r = p - (intercept + slope * t);
                    resid.push_back(r);
                }
            }
            double mean_r = 0.0; for (double r : resid) mean_r += r; mean_r /= resid.size();
            double m2 = 0.0, m3 = 0.0;
            for (double r : resid) {
                double d = r - mean_r;
                double d2 = d * d;
                m2 += d2;
                m3 += d2 * d;
            }
            m2 /= resid.size();
            m3 /= resid.size();
            if (m2 <= 0.0 || resid.size() < 3) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double g1 = m3 / std::pow(m2, 1.5);
            double n = static_cast<double>(resid.size());
            double adj = std::sqrt(n * (n - 1)) / (n - 2) * g1;
            res_ptr[i * d1 + j] = static_cast<float>(adj);
        }
    }
}

void bind_cube2mat_trend_resid_skew(py::module& m) {
    m.def("cube2mat_trend_resid_skew", &cube2mat_trend_resid_skew,
          py::arg("result"), py::arg("cubes_map"),
          "Adjusted skewness of residuals from linear trend of close over time.");
}

