#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_pac_ret_lag1_5(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            std::vector<float> r;
            r.reserve(d2-1);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = price_ptr[base + t - 1];
                float p1 = price_ptr[base + t];
                if (!(p0 > 0.f) || !(p1 > 0.f) || std::isnan(p0) || std::isnan(p1)) continue;
                r.push_back(std::log(p1 / p0));
            }
            ssize_t n = r.size();
            if (n < 2) { res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN(); continue; }
            int maxlag = static_cast<int>(std::min<ssize_t>(5, n - 1));
            std::vector<float> gamma(maxlag + 1, 0.f);
            float mean = 0.f;
            for (float val : r) mean += val;
            mean /= n;
            for (int lag = 0; lag <= maxlag; ++lag) {
                for (ssize_t t = lag; t < n; ++t) {
                    gamma[lag] += (r[t] - mean) * (r[t - lag] - mean);
                }
                gamma[lag] /= n;
            }
            if (!(gamma[0] > 0.f)) { res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<float> phi_old(maxlag + 1, 0.f), phi(maxlag + 1, 0.f);
            float var = gamma[0];
            int first = 0;
            float se = 1.96f / std::sqrt(static_cast<float>(n));
            for (int k = 1; k <= maxlag; ++k) {
                float acc = 0.f;
                for (int j2 = 1; j2 < k; ++j2) acc += phi_old[j2] * gamma[k - j2];
                float phi_k = (gamma[k] - acc) / var;
                phi[k] = phi_k;
                for (int j2 = 1; j2 < k; ++j2) {
                    phi[j2] = phi_old[j2] - phi_k * phi_old[k - j2];
                }
                var *= (1.f - phi_k * phi_k);
                if (first == 0 && std::fabs(phi_k) > se) first = k;
                phi_old = phi;
            }
            res_ptr[i*d1 + j] = static_cast<float>(first);
        }
    }
}

void bind_cube2mat_pac_ret_lag1_5(py::module& m) {
    m.def("cube2mat_pac_ret_lag1_5", &cube2mat_pac_ret_lag1_5,
          py::arg("result"), py::arg("cubes_map"),
          "First significant PACF lag (1..5) for log returns; 0 if none.");
}

