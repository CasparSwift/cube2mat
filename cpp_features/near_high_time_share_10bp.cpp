#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_near_high_time_share_10bp(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                        const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price") || !cubes_map.contains("interval_high")) {
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_high'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto high_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_high"]);
    auto pbuf = price_arr.request();
    auto hbuf = high_arr.request();
    if (pbuf.ndim != 3 || hbuf.ndim != 3) {
        throw std::runtime_error("inputs must be 3D arrays");
    }
    if (pbuf.shape[0] != hbuf.shape[0] || pbuf.shape[1] != hbuf.shape[1] || pbuf.shape[2] != hbuf.shape[2]) {
        throw std::runtime_error("arrays must have same shape");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    const float* high_ptr  = static_cast<float*>(hbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    const float thr = 10.0f / 10000.0f;

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            float H = -std::numeric_limits<float>::infinity();
            for (ssize_t t = 0; t < d2; ++t) {
                float h = high_ptr[base + t];
                if (std::isnan(h)) continue;
                if (h > H) H = h;
            }
            if (!(H > 0.0f)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            int cnt = 0;
            int tot = 0;
            for (ssize_t t = 0; t < d2; ++t) {
                float c = price_ptr[base + t];
                if (std::isnan(c)) continue;
                ++tot;
                float diff = H - c;
                if (diff < 0.0f) diff = 0.0f;
                if (H > 0.0f && diff / H <= thr) ++cnt;
            }
            res_ptr[i * d1 + j] = (tot > 0) ? static_cast<float>(cnt) / static_cast<float>(tot)
                                            : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_near_high_time_share_10bp(py::module& m) {
    m.def("cube2mat_near_high_time_share_10bp", &cube2mat_near_high_time_share_10bp,
          py::arg("result"), py::arg("cubes_map"),
          "Fraction of bars with close near session HIGH within 10bp.");
}

