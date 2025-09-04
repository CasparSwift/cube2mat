#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_early_vs_late_run_count_ratio(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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

    // early window: first 12 bars (09:30-10:29), returns indices 1..11
    const ssize_t early_ret_start = 1;
    const ssize_t early_ret_end = std::min<ssize_t>(12, d2 - 1); // exclusive upper bound
    // late window: last 12 bars (15:00-15:59), returns indices d2-11..d2-1
    const ssize_t late_ret_start = (d2 > 12) ? d2 - 11 : 1; // ensure at least 1
    const ssize_t late_ret_end = d2 - 1;

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            // compute log returns for entire day
            std::vector<int> sign(d2 - 1);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = close_ptr[base + t - 1];
                float p1 = close_ptr[base + t];
                float r = std::log(p1) - std::log(p0);
                if (!std::isfinite(r)) r = std::numeric_limits<float>::quiet_NaN();
                sign[t - 1] = (r > 0.0f) ? 1 : (r < 0.0f ? -1 : 0);
            }
            auto run_count = [&](ssize_t s, ssize_t e) -> float {
                int prev = 0;
                int cnt = 0;
                for (ssize_t k = s; k < e; ++k) {
                    int sg = sign[k];
                    if (sg == 0) continue;
                    if (cnt == 0 || sg != prev) {
                        ++cnt;
                        prev = sg;
                    }
                }
                return static_cast<float>(cnt);
            };
            float early = run_count(early_ret_start, early_ret_end);
            float late = run_count(late_ret_start, late_ret_end);
            if (!(late > 0.0f)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            } else {
                res_ptr[i * d1 + j] = early / late;
            }
        }
    }
}

void bind_cube2mat_early_vs_late_run_count_ratio(py::module& m) {
    m.def("cube2mat_early_vs_late_run_count_ratio", &cube2mat_early_vs_late_run_count_ratio,
          py::arg("result"), py::arg("cubes_map"),
          "Run count in 09:30-10:29 divided by run count in 15:00-15:59 (sign runs of log returns).");
}

