#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_near_close_ramp_slope_10m(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
    ssize_t start_idx = static_cast<ssize_t>(std::ceil(380.0 / interval_min));
    if (start_idx < 0) start_idx = 0;
    if (start_idx > d2 - 1) start_idx = d2 - 1;

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double sx=0.0, sy=0.0, sxx=0.0, sxy=0.0;
            int cnt=0;
            for (ssize_t t = start_idx; t < d2; ++t) {
                float c = price_ptr[base + t];
                if (std::isnan(c)) continue;
                double x = (t - start_idx) * interval_min;
                double y = static_cast<double>(c);
                sx += x; sy += y; sxx += x*x; sxy += x*y; ++cnt;
            }
            if (cnt >= 2) {
                double den = sxx - sx*sx/cnt;
                if (den > 0.0) {
                    double beta = (sxy - sx*sy/cnt) / den;
                    res_ptr[i * d1 + j] = static_cast<float>(beta);
                    continue;
                }
            }
            res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_near_close_ramp_slope_10m(py::module& m) {
    m.def("cube2mat_near_close_ramp_slope_10m", &cube2mat_near_close_ramp_slope_10m,
          py::arg("result"), py::arg("cubes_map"),
          "Tail 10-minute linear slope of close vs time.");
}

