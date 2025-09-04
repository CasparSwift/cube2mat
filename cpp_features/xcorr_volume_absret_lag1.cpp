#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_xcorr_volume_absret_lag1(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                      const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price") || !cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_volume'");
    }
    auto price_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto volume_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto pbuf = price_arr.request();
    auto vbuf = volume_arr.request();
    if (pbuf.ndim != 3 || vbuf.ndim != 3) {
        throw std::runtime_error("inputs must be 3D arrays");
    }
    if (pbuf.shape[0] != vbuf.shape[0] || pbuf.shape[1] != vbuf.shape[1] || pbuf.shape[2] != vbuf.shape[2]) {
        throw std::runtime_error("last_price and interval_volume must have same shape");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* price_ptr  = static_cast<float*>(pbuf.ptr);
    const float* volume_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double sx=0.0, sy=0.0, sxx=0.0, syy=0.0, sxy=0.0; int cnt=0;
            for (ssize_t t = 0; t < d2 - 1; ++t) {
                float v = volume_ptr[base + t];
                float p0 = price_ptr[base + t];
                float p1 = price_ptr[base + t + 1];
                if (std::isnan(v) || std::isnan(p0) || std::isnan(p1) || !(p0 > 0.f) || !(p1 > 0.f))
                    continue;
                float r = std::log(p1) - std::log(p0);
                float y = std::fabs(r);
                if (std::isnan(y)) continue;
                double x = v;
                sx += x; sy += y; sxx += x*x; syy += y*y; sxy += x*y; ++cnt;
            }
            if (cnt > 2) {
                double cov = sxy - sx*sy/cnt;
                double varx = sxx - sx*sx/cnt;
                double vary = syy - sy*sy/cnt;
                res_ptr[i*d1 + j] = (varx>0.0 && vary>0.0) ? static_cast<float>(cov/std::sqrt(varx*vary))
                                                            : std::numeric_limits<float>::quiet_NaN();
            } else {
                res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_xcorr_volume_absret_lag1(py::module& m) {
    m.def("cube2mat_xcorr_volume_absret_lag1", &cube2mat_xcorr_volume_absret_lag1,
          py::arg("result"), py::arg("cubes_map"),
          "Corr(volume_t, |logret_{t+1}|) in RTH; lead-lag (volume leads).");
}

