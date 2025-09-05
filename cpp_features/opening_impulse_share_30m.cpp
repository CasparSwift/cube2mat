#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_opening_impulse_share_30m(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                        const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];

    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1)
        throw std::runtime_error("result must be (d0,d1)");

    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i*(d1*d2) + j*d2;
            float p0 = price_ptr[base];
            float plast = price_ptr[base + d2 - 1];
            if (!(p0 > 0.f) || !(plast > 0.f) || std::isnan(p0) || std::isnan(plast)) {
                res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            float total = std::log(plast / p0);
            if (!std::isfinite(total) || total == 0.f) {
                res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            ssize_t end = std::min<ssize_t>(6, d2 - 1);
            float sum = 0.f;
            for (ssize_t t = 1; t <= end; ++t) {
                float pp = price_ptr[base + t - 1];
                float pc = price_ptr[base + t];
                if (!(pp > 0.f) || !(pc > 0.f) || std::isnan(pp) || std::isnan(pc)) continue;
                sum += std::log(pc / pp);
            }
            res_ptr[i*d1 + j] = sum / total;
        }
    }
}

void bind_cube2mat_opening_impulse_share_30m(py::module& m) {
    m.def("cube2mat_opening_impulse_share_30m", &cube2mat_opening_impulse_share_30m,
          py::arg("result"), py::arg("cubes_map"),
          "Sum of log returns in 09:30–09:59 divided by total O→C log return.");
}

