#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static float higuchi_fd(const std::vector<float>& x, int kmax) {
    int N = static_cast<int>(x.size());
    if (N < kmax + 2) return std::numeric_limits<float>::quiet_NaN();
    std::vector<double> Lk; Lk.reserve(kmax-1);
    std::vector<double> K;  K.reserve(kmax-1);
    for (int k = 2; k <= kmax; ++k) {
        std::vector<double> Lm;
        for (int m = 0; m < k; ++m) {
            std::vector<double> idx;
            for (int t = m; t < N; t += k) idx.push_back(t);
            if (idx.size() < 2) continue;
            double L = 0.0;
            for (size_t t = 1; t < idx.size(); ++t) {
                double diff = std::fabs(x[(size_t)idx[t]] - x[(size_t)idx[t-1]]);
                L += diff;
            }
            L = L * (N - 1) / ((idx.size() - 1) * k);
            Lm.push_back(L);
        }
        if (!Lm.empty()) {
            double sum = 0.0; for (double v : Lm) sum += v; sum /= Lm.size();
            Lk.push_back(sum); K.push_back(1.0 / k);
        }
    }
    if (Lk.size() < 2) return std::numeric_limits<float>::quiet_NaN();
    double sumx=0.0,sumy=0.0,sumxx=0.0,sumxy=0.0; size_t n=Lk.size();
    for (size_t i=0;i<n;++i){
        double x1=std::log(K[i]);
        double y1=std::log(Lk[i]);
        sumx+=x1; sumy+=y1; sumxx+=x1*x1; sumxy+=x1*y1;
    }
    double denom = n*sumxx - sumx*sumx;
    if (denom==0.0) return std::numeric_limits<float>::quiet_NaN();
    double slope = (n*sumxy - sumx*sumy) / denom;
    return static_cast<float>(slope);
}

void cube2mat_higuchi_fd_logret_kmax7(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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

    const int kmax = 7;
    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> r;
            r.reserve(d2 - 1);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = close_ptr[base + t - 1];
                float p1 = close_ptr[base + t];
                float ret = std::log(p1) - std::log(p0);
                if (std::isfinite(ret)) r.push_back(ret);
            }
            res_ptr[i * d1 + j] = higuchi_fd(r, kmax);
        }
    }
}

void bind_cube2mat_higuchi_fd_logret_kmax7(py::module& m) {
    m.def("cube2mat_higuchi_fd_logret_kmax7", &cube2mat_higuchi_fd_logret_kmax7,
          py::arg("result"), py::arg("cubes_map"),
          "Higuchi fractal dimension estimate of log returns (k_max=7) within RTH.");
}

