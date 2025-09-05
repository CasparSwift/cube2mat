#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_open_15m_absret_share(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
    ssize_t k = static_cast<ssize_t>(std::ceil(15.0 / interval_min));
    if (k < 1) k = 1;

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> r;
            r.reserve(d2);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = price_ptr[base + t - 1];
                float p1 = price_ptr[base + t];
                if (!(p0 > 0.f) || !(p1 > 0.f) || std::isnan(p0) || std::isnan(p1)) continue;
                double val = static_cast<double>(p1) / p0 - 1.0;
                if (std::isfinite(val)) r.push_back(val);
            }
            const size_t n = r.size();
            if (n == 0) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double denom = 0.0;
            for (double v : r) denom += std::fabs(v);
            if (!(denom > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            ssize_t kk = std::min<ssize_t>(k, n);
            double num = 0.0;
            for (ssize_t t = 0; t < kk; ++t) num += std::fabs(r[t]);
            res_ptr[i * d1 + j] = static_cast<float>(num / denom);
        }
    }
}

void bind_cube2mat_open_15m_absret_share(py::module& m) {
    m.def("cube2mat_open_15m_absret_share", &cube2mat_open_15m_absret_share,
          py::arg("result"), py::arg("cubes_map"),
          "Share of absolute returns in first 15 minutes; ret by close.pct_change.");
}

