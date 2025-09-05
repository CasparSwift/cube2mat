#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_time_to_half_openclose_ret_frac(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                              const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }

    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];

    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }

    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> rets;
            rets.reserve(d2);
            float prev = std::numeric_limits<float>::quiet_NaN();
            for (ssize_t t = 0; t < d2; ++t) {
                float c = price_ptr[base + t];
                if (std::isnan(c)) { prev = std::numeric_limits<float>::quiet_NaN(); continue; }
                if (!std::isnan(prev) && prev > 0.0f && c > 0.0f) {
                    double r = std::log(static_cast<double>(c) / prev);
                    rets.push_back(r);
                }
                prev = c;
            }
            size_t n = rets.size();
            if (n == 0) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double total = 0.0;
            for (double r : rets) total += r;
            if (total == 0.0) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double target = 0.5 * std::fabs(total);
            double sign = (total > 0.0) ? 1.0 : -1.0;
            double csum = 0.0;
            ssize_t idx = -1;
            for (size_t t = 0; t < n; ++t) {
                csum += rets[t];
                if (sign * csum >= target) { idx = static_cast<ssize_t>(t); break; }
            }
            if (idx >= 0) {
                res_ptr[i * d1 + j] = static_cast<float>((idx + 1.0) / n);
            } else {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_time_to_half_openclose_ret_frac(py::module& m) {
    m.def("cube2mat_time_to_half_openclose_ret_frac", &cube2mat_time_to_half_openclose_ret_frac,
          py::arg("result"), py::arg("cubes_map"),
          "Fraction of returns elapsed when cumulative log return first reaches half of O→C total.");
}

