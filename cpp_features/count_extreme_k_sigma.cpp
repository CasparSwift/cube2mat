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

// Count of bars with |ret| > k * sigma_robust (sigma from IQR)
void cube2mat_count_extreme_k_sigma(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                    const py::dict& cubes_map,
                                    float k) {
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

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> rets; rets.reserve(d2);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = price_ptr[base + t - 1];
                float p1 = price_ptr[base + t];
                if (!(p0 > 0.f) || !(p1 > 0.f) || std::isnan(p0) || std::isnan(p1)) continue;
                float r = (p1 - p0) / p0;
                if (std::isfinite(r)) rets.push_back(r);
            }
            if (rets.size() < 2) {
                res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::sort(rets.begin(), rets.end());
            size_t n = rets.size();
            float q1 = rets[static_cast<size_t>(0.25 * (n - 1))];
            float q3 = rets[static_cast<size_t>(0.75 * (n - 1))];
            float iqr = q3 - q1;
            float sigma = iqr / 1.349f;
            if (!(sigma > 0.f)) {
                res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            float thresh = k * sigma;
            int cnt = 0;
            for (float r : rets) {
                if (std::fabs(r) > thresh) ++cnt;
            }
            res_ptr[i*d1 + j] = static_cast<float>(cnt);
        }
    }
}

void bind_cube2mat_count_extreme_k_sigma(py::module& m) {
    m.def("cube2mat_count_extreme_k_sigma", &cube2mat_count_extreme_k_sigma,
          py::arg("result"), py::arg("cubes_map"), py::arg("k") = 3.0f,
          "Count of bars with |ret| exceeding k·sigma_robust (IQR-based) within RTH.");
}

