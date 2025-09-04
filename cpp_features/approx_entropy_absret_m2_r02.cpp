#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static float apen(const std::vector<float>& x, int m, float r) {
    const int n = static_cast<int>(x.size());
    if (n <= m + 1 || !(r > 0.0f)) return std::numeric_limits<float>::quiet_NaN();
    auto phi = [&](int mm) {
        int N = n - mm + 1;
        std::vector<double> C(N, 0.0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double d = 0.0;
                for (int k = 0; k < mm; ++k) {
                    double diff = std::fabs(x[i + k] - x[j + k]);
                    if (diff > d) d = diff;
                }
                if (d <= r) C[i] += 1.0;
            }
            C[i] /= N;
            if (C[i] <= 0.0) C[i] = std::numeric_limits<double>::quiet_NaN();
            else C[i] = std::log(C[i]);
        }
        double sum = 0.0;
        int cnt = 0;
        for (double v : C) if (std::isfinite(v)) { sum += v; ++cnt; }
        return (cnt > 0) ? static_cast<double>(sum / cnt) : std::numeric_limits<double>::quiet_NaN();
    };
    double p1 = phi(m);
    double p2 = phi(m + 1);
    if (!std::isfinite(p1) || !std::isfinite(p2)) return std::numeric_limits<float>::quiet_NaN();
    return static_cast<float>(p1 - p2);
}

void cube2mat_approx_entropy_absret_m2_r02(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                           const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto close_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto cbuf = close_arr.request();
    if (cbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array");
    }
    const ssize_t d0 = cbuf.shape[0];
    const ssize_t d1 = cbuf.shape[1];
    const ssize_t d2 = cbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D (d0,d1)");
    }
    const float* close_ptr = static_cast<float*>(cbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> absr;
            absr.reserve(d2 - 1);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = close_ptr[base + t - 1];
                float p1 = close_ptr[base + t];
                float r = std::log(p1) - std::log(p0);
                if (std::isfinite(r)) absr.push_back(std::fabs(r));
            }
            if (absr.size() <= 5) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double mean = 0.0;
            for (float v : absr) mean += v; mean /= absr.size();
            double var = 0.0; for (float v : absr) var += (v - mean)*(v - mean); var /= absr.size();
            double sd = std::sqrt(var);
            float rparm = static_cast<float>(0.2 * sd);
            res_ptr[i * d1 + j] = apen(absr, 2, rparm);
        }
    }
}

void bind_cube2mat_approx_entropy_absret_m2_r02(py::module& m) {
    m.def("cube2mat_approx_entropy_absret_m2_r02", &cube2mat_approx_entropy_absret_m2_r02,
          py::arg("result"), py::arg("cubes_map"),
          "Approximate Entropy of |log returns| with m=2, r=0.2·std(|r|) within RTH.");
}

