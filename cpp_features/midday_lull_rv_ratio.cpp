#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_midday_lull_rv_ratio(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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

    const double TOTAL_MIN = 389.0;
    const double interval_min = TOTAL_MIN / static_cast<double>(d2 - 1);
    ssize_t start_idx = static_cast<ssize_t>(std::ceil(150.0 / interval_min));
    ssize_t end_idx   = static_cast<ssize_t>(std::floor(205.0 / interval_min));
    if (start_idx < 1) start_idx = 1;
    if (end_idx > d2 - 1) end_idx = d2 - 1;

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double total = 0.0;
            double mid = 0.0;
            int cnt_total = 0;
            int cnt_mid = 0;
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = price_ptr[base + t - 1];
                float p1 = price_ptr[base + t];
                if (!(p0 > 0.f) || !(p1 > 0.f) || std::isnan(p0) || std::isnan(p1)) continue;
                double r = std::log(static_cast<double>(p1)) - std::log(static_cast<double>(p0));
                if (!std::isfinite(r)) continue;
                double r2 = r * r;
                total += r2;
                ++cnt_total;
                if (t >= start_idx && t <= end_idx) {
                    mid += r2;
                    ++cnt_mid;
                }
            }
            if (cnt_total >= 3 && total > 0.0 && cnt_mid > 0) {
                double ratio = (mid / total) / (static_cast<double>(cnt_mid) / cnt_total);
                res_ptr[i * d1 + j] = static_cast<float>(ratio);
            } else {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_midday_lull_rv_ratio(py::module& m) {
    m.def("cube2mat_midday_lull_rv_ratio", &cube2mat_midday_lull_rv_ratio,
          py::arg("result"), py::arg("cubes_map"),
          "RV density at midday vs session average.");
}

