#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_xcorr_sign_volume_leadlag_peak(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
    const int K = 5;

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> sgn(d2, std::numeric_limits<float>::quiet_NaN());
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = price_ptr[base + t - 1];
                float p1 = price_ptr[base + t];
                if (!(p0 > 0.f) || !(p1 > 0.f) || std::isnan(p0) || std::isnan(p1)) continue;
                float ret = (p1 - p0) / p0;
                if (ret > 0) sgn[t] = 1.f;
                else if (ret < 0) sgn[t] = -1.f;
                else sgn[t] = 0.f;
            }
            float best = std::numeric_limits<float>::quiet_NaN();
            for (int lag = -K; lag <= K; ++lag) {
                ssize_t start_x, start_y, len;
                if (lag < 0) {
                    start_x = -lag;
                    start_y = 0;
                    len = d2 + lag;
                } else if (lag > 0) {
                    start_x = 0;
                    start_y = lag;
                    len = d2 - lag;
                } else {
                    start_x = 0;
                    start_y = 0;
                    len = d2;
                }
                start_x += 1;
                start_y += 1;
                len -= 1;
                if (len < 3 || start_x + len > d2 || start_y + len > d2) continue;
                double sx=0.0, sy=0.0, sxx=0.0, syy=0.0, sxy=0.0; int cnt=0;
                for (ssize_t t = 0; t < len; ++t) {
                    float x = sgn[start_x + t];
                    float y = volume_ptr[base + start_y + t];
                    if (std::isnan(x) || std::isnan(y)) continue;
                    sx += x; sy += y; sxx += x*x; syy += y*y; sxy += x*y; ++cnt;
                }
                if (cnt < 3) continue;
                double cov = sxy - sx*sy/cnt;
                double varx = sxx - sx*sx/cnt;
                double vary = syy - sy*sy/cnt;
                if (varx <= 0.0 || vary <= 0.0) continue;
                double c = cov / std::sqrt(varx * vary);
                if (!std::isnan(c) && (std::isnan(best) || std::fabs(c) > std::fabs(best))) {
                    best = static_cast<float>(c);
                }
            }
            res_ptr[i*d1 + j] = best;
        }
    }
}

void bind_cube2mat_xcorr_sign_volume_leadlag_peak(py::module& m) {
    m.def("cube2mat_xcorr_sign_volume_leadlag_peak", &cube2mat_xcorr_sign_volume_leadlag_peak,
          py::arg("result"), py::arg("cubes_map"),
          "Peak |corr(sign(ret), volume shift)| over lags [-5,5] in RTH.");
}

