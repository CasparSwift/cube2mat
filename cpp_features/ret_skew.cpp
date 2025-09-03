#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

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

void bind_ret_skew(py::module& m) {
    m.def("cube2mat_ret_skew", &cube2mat_ret_skew,
          py::arg("result"), py::arg("cubes_map"),
          "Sample-adjusted skewness of intraday log returns.");
}
