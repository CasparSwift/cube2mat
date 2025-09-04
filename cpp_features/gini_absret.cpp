#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static float gini_coeff(std::vector<float>& vals) {
    const size_t n = vals.size();
    if (n == 0) return std::numeric_limits<float>::quiet_NaN();
    std::sort(vals.begin(), vals.end());
    double sum = 0.0;
    double cumsum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += vals[i];
        cumsum += static_cast<double>(i + 1) * vals[i];
    }
    if (sum <= 0.0) return std::numeric_limits<float>::quiet_NaN();
    double g = (2.0 * cumsum) / (n * sum) - (static_cast<double>(n + 1)) / n;
    return static_cast<float>(g);
}

void cube2mat_gini_absret(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<float> vals;
            vals.reserve(d2);
            for (ssize_t t = 1; t < d2; ++t) {
                float p0 = price_ptr[base + t - 1];
                float p1 = price_ptr[base + t];
                if (p0 > 0.0f && p1 > 0.0f && !std::isnan(p0) && !std::isnan(p1)) {
                    vals.push_back(std::fabs(std::log(p1 / p0)));
                }
            }
            res_ptr[i * d1 + j] = gini_coeff(vals);
        }
    }
}

void bind_cube2mat_gini_absret(py::module& m) {
    m.def("cube2mat_gini_absret", &cube2mat_gini_absret,
          py::arg("result"), py::arg("cubes_map"),
          "Gini concentration of absolute log returns during RTH.");
}

