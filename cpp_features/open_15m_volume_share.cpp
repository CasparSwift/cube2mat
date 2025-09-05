#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_open_15m_volume_share(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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

    const double TOTAL_MIN = 389.0;
    const double interval_min = TOTAL_MIN / static_cast<double>(d2);
    ssize_t k = static_cast<ssize_t>(std::ceil(15.0 / interval_min));
    if (k < 1) k = 1;

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> v;
            v.reserve(d2);
            double total = 0.0;
            for (ssize_t t = 0; t < d2; ++t) {
                float vv = vol_ptr[base + t];
                if (std::isnan(vv)) continue;
                v.push_back(static_cast<double>(vv));
                total += vv;
            }
            const size_t n = v.size();
            if (n == 0 || !(total > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            ssize_t kk = std::min<ssize_t>(k, n);
            double num = 0.0;
            for (ssize_t t = 0; t < kk; ++t) num += v[t];
            res_ptr[i * d1 + j] = static_cast<float>(num / total);
        }
    }
}

void bind_cube2mat_open_15m_volume_share(py::module& m) {
    m.def("cube2mat_open_15m_volume_share", &cube2mat_open_15m_volume_share,
          py::arg("result"), py::arg("cubes_map"),
          "Share of volume in first 15 minutes; sum(vol in t<15) / sum(vol all).");
}

