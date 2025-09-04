#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_entropy_sign_ret(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            int counts[3] = {0,0,0}; // index 0->-1,1->0,2->+1
            int total = 0;
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = close_ptr[base + t - 1];
                float p1 = close_ptr[base + t];
                float r = (p1 - p0) / p0;
                if (!std::isfinite(r)) continue;
                int s = (r > 0.0f) ? 2 : (r < 0.0f ? 0 : 1);
                counts[s]++;
                ++total;
            }
            int k = 0; double H=0.0;
            for (int idx=0; idx<3; ++idx) {
                if (counts[idx] > 0) {
                    double p = static_cast<double>(counts[idx]) / total;
                    H -= p * std::log(p);
                    ++k;
                }
            }
            if (total < 1 || k < 2) {
                res_ptr[i*d1+j] = std::numeric_limits<float>::quiet_NaN();
            } else {
                res_ptr[i*d1+j] = static_cast<float>(H / std::log(static_cast<double>(k)));
            }
        }
    }
}

void bind_cube2mat_entropy_sign_ret(py::module& m) {
    m.def("cube2mat_entropy_sign_ret", &cube2mat_entropy_sign_ret,
          py::arg("result"), py::arg("cubes_map"),
          "Normalized entropy (0..1) of simple-return sign distribution.");
}

