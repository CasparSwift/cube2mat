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

void cube2mat_var_conc_top_decile(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            std::vector<float> sqr;
            sqr.reserve(d2);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = price_ptr[base + t - 1];
                float p1 = price_ptr[base + t];
                if (p0 > 0.f && p1 > 0.f && !std::isnan(p0) && !std::isnan(p1)) {
                    float r = std::log(p1 / p0);
                    sqr.push_back(r*r);
                }
            }
            size_t n = sqr.size();
            if (n == 0) {
                res_ptr[i*d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::sort(sqr.begin(), sqr.end(), std::greater<float>());
            size_t topn = std::max<size_t>(1, static_cast<size_t>(std::ceil(0.1 * n)));
            double total = 0.0; for (float v: sqr) total += v;
            double top = 0.0; for (size_t k=0; k<topn && k<n; ++k) top += sqr[k];
            res_ptr[i*d1 + j] = static_cast<float>((total>0.0)? top/total : std::numeric_limits<float>::quiet_NaN());
        }
    }
}

void bind_cube2mat_var_conc_top_decile(py::module& m) {
    m.def("cube2mat_var_conc_top_decile", &cube2mat_var_conc_top_decile,
          py::arg("result"), py::arg("cubes_map"),
          "Share of realized variance contributed by top 10% r^2.");
}

