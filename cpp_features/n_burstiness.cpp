#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_n_burstiness(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                           const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_n")) {
        throw std::runtime_error("cubes_map must contain 'interval_n'");
    }
    auto n_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_n"]);
    auto nbuf = n_arr.request();
    if (nbuf.ndim != 3) {
        throw std::runtime_error("interval_n must be 3D");
    }
    const ssize_t d0 = nbuf.shape[0];
    const ssize_t d1 = nbuf.shape[1];
    const ssize_t d2 = nbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }
    const float* n_ptr = static_cast<float*>(nbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            double sum = 0.0;
            double sumsq = 0.0;
            int cnt = 0;
            for (ssize_t t = 0; t < d2; ++t) {
                float v = n_ptr[base + t];
                if (std::isnan(v)) continue;
                double dv = static_cast<double>(v);
                sum += dv;
                sumsq += dv * dv;
                ++cnt;
            }
            if (cnt >= 2) {
                double mean = sum / cnt;
                double var = (sumsq - sum * sum / cnt) / (cnt - 1);
                if (mean > 0.0 && var > 0.0) {
                    double sd = std::sqrt(var);
                    res_ptr[i * d1 + j] = static_cast<float>((sd - mean) / (sd + mean));
                    continue;
                }
            }
            res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_n_burstiness(py::module& m) {
    m.def("cube2mat_n_burstiness", &cube2mat_n_burstiness,
          py::arg("result"), py::arg("cubes_map"),
          "Burstiness of trade counts n across intraday bars: (std-mean)/(std+mean).");
}

