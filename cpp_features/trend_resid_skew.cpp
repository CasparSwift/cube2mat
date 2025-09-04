#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <limits>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_trend_resid_skew(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                               const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array (d0,d1,d2)");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1) and match last_price leading dims");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> y;
            y.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float v = price_ptr[base + t];
                if (std::isfinite(v)) y.push_back(v);
            }
            const ssize_t n = static_cast<ssize_t>(y.size());
            if (n < 3) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double S_t = 0.0, S_y = 0.0, S_tt = 0.0, S_ty = 0.0;
            for (ssize_t t = 0; t < n; ++t) {
                double tt = static_cast<double>(t) / (n - 1);
                double yy = y[t];
                S_t += tt;
                S_y += yy;
                S_tt += tt * tt;
                S_ty += tt * yy;
            }
            double denom = static_cast<double>(n) * S_tt - S_t * S_t;
            if (denom == 0.0) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double beta1 = (static_cast<double>(n) * S_ty - S_t * S_y) / denom;
            double beta0 = (S_y - beta1 * S_t) / static_cast<double>(n);
            std::vector<double> e(n);
            for (ssize_t t = 0; t < n; ++t) {
                double tt = static_cast<double>(t) / (n - 1);
                e[t] = y[t] - (beta0 + beta1 * tt);
            }
            double m = 0.0;
            for (double v : e) m += v;
            m /= static_cast<double>(n);
            double c2 = 0.0, c3 = 0.0;
            for (double v : e) {
                double d = v - m;
                double d2 = d * d;
                c2 += d2;
                c3 += d2 * d;
            }
            c2 /= static_cast<double>(n);
            c3 /= static_cast<double>(n);
            if (!(c2 > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double g1 = c3 / std::pow(c2, 1.5);
            double adj = std::sqrt(static_cast<double>(n) * (n - 1.0)) / (n - 2.0);
            double val = adj * g1;
            res_ptr[i * d1 + j] = std::isfinite(val) ? static_cast<float>(val)
                                                     : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_trend_resid_skew(py::module& m) {
    m.def("cube2mat_trend_resid_skew", &cube2mat_trend_resid_skew,
          py::arg("result"), py::arg("cubes_map"),
          "Adjusted skewness of trend residuals.");
}
