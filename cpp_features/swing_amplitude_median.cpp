#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static float median(std::vector<float>& v) {
    if (v.empty()) return std::numeric_limits<float>::quiet_NaN();
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) return v[n/2];
    return 0.5f * (v[n/2 - 1] + v[n/2]);
}

void cube2mat_swing_amplitude_median(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                     const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto close_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto cbuf = close_arr.request();
    if (cbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array");
    }
    const ssize_t d0 = cbuf.shape[0];
    const ssize_t d1 = cbuf.shape[1];
    const ssize_t d2 = cbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1)");
    }
    const float* close_ptr = static_cast<float*>(cbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> amps;
            float sum = 0.0f;
            int cur_sign = 0;
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = close_ptr[base + t - 1];
                float p1 = close_ptr[base + t];
                float r = std::log(p1) - std::log(p0);
                if (!std::isfinite(r)) continue;
                int s = (r > 0.0f) ? 1 : (r < 0.0f ? -1 : 0);
                if (s == 0) {
                    if (cur_sign != 0) {
                        amps.push_back(std::fabs(sum));
                        cur_sign = 0; sum = 0.0f;
                    }
                    continue;
                }
                if (cur_sign == 0) {
                    cur_sign = s; sum = r;
                } else if (s == cur_sign) {
                    sum += r;
                } else {
                    amps.push_back(std::fabs(sum));
                    cur_sign = s; sum = r;
                }
            }
            if (cur_sign != 0) amps.push_back(std::fabs(sum));
            res_ptr[i * d1 + j] = median(amps);
        }
    }
}

void bind_cube2mat_swing_amplitude_median(py::module& m) {
    m.def("cube2mat_swing_amplitude_median", &cube2mat_swing_amplitude_median,
          py::arg("result"), py::arg("cubes_map"),
          "Median absolute swing amplitude per sign run (sum of returns over the run, absolute value).");
}

