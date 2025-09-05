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

static float quantile_sorted(const std::vector<float>& vals, float q) {
    const size_t n = vals.size();
    if (n == 0) return std::numeric_limits<float>::quiet_NaN();
    if (n == 1) return vals[0];
    double idx = (n - 1) * q;
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    double h = idx - lo;
    return static_cast<float>((1.0 - h) * vals[lo] + h * vals[hi]);
}

void cube2mat_iqr_vol_logret(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> rets;
            rets.reserve(d2);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = price_ptr[base + t - 1];
                float p1 = price_ptr[base + t];
                if (p0 > 0.f && p1 > 0.f && !std::isnan(p0) && !std::isnan(p1))
                    rets.push_back(std::log(p1 / p0));
            }
            if (rets.empty()) {
                res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::sort(rets.begin(), rets.end());
            float q1 = quantile_sorted(rets, 0.25f);
            float q3 = quantile_sorted(rets, 0.75f);
            float iqr = q3 - q1;
            res_ptr[i*d1 + j] = 0.7413f * iqr;
        }
    }
}

void bind_cube2mat_iqr_vol_logret(py::module& m) {
    m.def("cube2mat_iqr_vol_logret", &cube2mat_iqr_vol_logret,
          py::arg("result"), py::arg("cubes_map"),
          "Robust volatility via 0.7413*IQR of log returns.");
}

