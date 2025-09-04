#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static void detrend(const std::vector<double>& y, std::vector<double>& e) {
    size_t n = y.size();
    double sum_t = 0.0, sum_y = 0.0, sum_t2 = 0.0, sum_ty = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / (n - 1);
        double yy = y[i];
        sum_t += t;
        sum_y += yy;
        sum_t2 += t * t;
        sum_ty += t * yy;
    }
    double den = n * sum_t2 - sum_t * sum_t;
    double a = (sum_y * sum_t2 - sum_t * sum_ty) / den;
    double b = (n * sum_ty - sum_t * sum_y) / den;
    e.resize(n);
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / (n - 1);
        e[i] = y[i] - (a + b * t);
        mean += e[i];
    }
    mean /= n;
    for (size_t i = 0; i < n; ++i) e[i] -= mean;
}

void cube2mat_spectral_lowfreq_share_close(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                           const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    const double p = 0.10;

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> y;
            y.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float c = price_ptr[base + t];
                if (!std::isnan(c)) y.push_back(static_cast<double>(c));
            }
            size_t n = y.size();
            if (n < 8) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::vector<double> e;
            detrend(y, e);
            // FFT via naive DFT
            size_t m = e.size();
            if (m <= 1) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            size_t pos_m = m - 1; // positive freq count
            std::vector<double> pow(pos_m);
            for (size_t k = 1; k < m; ++k) {
                double re = 0.0, im = 0.0;
                for (size_t t = 0; t < m; ++t) {
                    double ang = 2.0 * M_PI * k * t / m;
                    re += e[t] * std::cos(ang);
                    im -= e[t] * std::sin(ang);
                }
                pow[k-1] = re * re + im * im;
            }
            double total = 0.0;
            for (double v : pow) total += v;
            if (total <= 0.0 || !std::isfinite(total)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            size_t mpos = pow.size();
            size_t kcut = std::max<size_t>(1, static_cast<size_t>(std::floor(p * mpos)));
            double low = 0.0;
            for (size_t k = 0; k < kcut && k < mpos; ++k) low += pow[k];
            res_ptr[i * d1 + j] = static_cast<float>(low / total);
        }
    }
}

void bind_cube2mat_spectral_lowfreq_share_close(py::module& m) {
    m.def("cube2mat_spectral_lowfreq_share_close", &cube2mat_spectral_lowfreq_share_close,
          py::arg("result"), py::arg("cubes_map"),
          "Share of low-frequency FFT energy of detrended close (p=10%).");
}

