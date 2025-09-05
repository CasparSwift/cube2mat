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

void cube2mat_theil_sen_slope_close_time(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                         const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }

    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];

    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }

    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> times;
            std::vector<double> prices;
            times.reserve(d2);
            prices.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float c = price_ptr[base + t];
                if (std::isnan(c)) continue;
                times.push_back(static_cast<double>(t) * 5.0);
                prices.push_back(static_cast<double>(c));
            }
            const size_t n = prices.size();
            if (n < 2) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::vector<double> slopes;
            slopes.reserve(n * (n - 1) / 2);
            for (size_t a = 0; a < n; ++a) {
                for (size_t b = a + 1; b < n; ++b) {
                    double dt = times[b] - times[a];
                    if (dt == 0.0) continue;
                    slopes.push_back((prices[b] - prices[a]) / dt);
                }
            }
            if (slopes.empty()) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            size_t m = slopes.size();
            std::nth_element(slopes.begin(), slopes.begin() + m / 2, slopes.end());
            double median = slopes[m / 2];
            if (m % 2 == 0) {
                std::nth_element(slopes.begin(), slopes.begin() + m / 2 - 1, slopes.end());
                median = 0.5 * (median + slopes[m / 2 - 1]);
            }
            res_ptr[i * d1 + j] = static_cast<float>(median);
        }
    }
}

void bind_cube2mat_theil_sen_slope_close_time(py::module& m) {
    m.def("cube2mat_theil_sen_slope_close_time", &cube2mat_theil_sen_slope_close_time,
          py::arg("result"), py::arg("cubes_map"),
          "Robust Theil–Sen slope of close vs time (minutes).");
}

