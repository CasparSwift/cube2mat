#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_time_to_20pct_volume_frac(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                        const py::dict& cubes_map) {
    if (!cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'interval_volume'");
    }

    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf = vol_arr.request();
    if (vbuf.ndim != 3) {
        throw std::runtime_error("interval_volume must be 3D");
    }
    const ssize_t d0 = vbuf.shape[0];
    const ssize_t d1 = vbuf.shape[1];
    const ssize_t d2 = vbuf.shape[2];

    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }

    const float* vol_ptr = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> vols;
            vols.reserve(d2);
            double total = 0.0;
            for (ssize_t t = 0; t < d2; ++t) {
                float v = vol_ptr[base + t];
                if (std::isnan(v)) continue;
                vols.push_back(static_cast<double>(v));
                total += v;
            }
            size_t n = vols.size();
            if (n == 0 || !(total > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double thr = 0.2 * total;
            double csum = 0.0;
            ssize_t k = -1;
            for (size_t t = 0; t < n; ++t) {
                csum += vols[t];
                if (csum >= thr) { k = static_cast<ssize_t>(t); break; }
            }
            if (k >= 0) {
                res_ptr[i * d1 + j] = static_cast<float>((k + 1.0) / n);
            } else {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_time_to_20pct_volume_frac(py::module& m) {
    m.def("cube2mat_time_to_20pct_volume_frac", &cube2mat_time_to_20pct_volume_frac,
          py::arg("result"), py::arg("cubes_map"),
          "Fraction of session minutes to reach 20% cumulative volume.");
}

