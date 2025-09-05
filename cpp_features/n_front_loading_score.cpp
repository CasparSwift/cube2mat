#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_n_front_loading_score(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            double total = 0.0;
            for (ssize_t t = 0; t < d2; ++t) {
                float v = n_ptr[base + t];
                if (std::isnan(v)) continue;
                total += static_cast<double>(v);
            }
            if (!(total > 0.0)) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::vector<double> xs;
            std::vector<double> ys;
            xs.reserve(d2);
            ys.reserve(d2);
            double cum = 0.0;
            for (ssize_t t = 0; t < d2; ++t) {
                float v = n_ptr[base + t];
                if (std::isnan(v)) continue;
                double dv = static_cast<double>(v);
                cum += dv;
                double tf = (d2 > 1) ? static_cast<double>(t) / static_cast<double>(d2 - 1) : 0.0;
                xs.push_back(tf);
                ys.push_back(cum / total);
            }
            if (xs.size() < 2) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            if (xs.front() > 0.0) {
                xs.insert(xs.begin(), 0.0);
                ys.insert(ys.begin(), 0.0);
            }
            if (xs.back() < 1.0) {
                xs.push_back(1.0);
                ys.push_back(1.0);
            }
            double auc = 0.0;
            for (size_t k = 1; k < xs.size(); ++k) {
                double dx = xs[k] - xs[k - 1];
                double av = (ys[k] + ys[k - 1]) / 2.0;
                auc += av * dx;
            }
            double score = 2.0 * auc - 1.0;
            if (score > 1.0) score = 1.0;
            if (score < -1.0) score = -1.0;
            res_ptr[i * d1 + j] = static_cast<float>(score);
        }
    }
}

void bind_cube2mat_n_front_loading_score(py::module& m) {
    m.def("cube2mat_n_front_loading_score", &cube2mat_n_front_loading_score,
          py::arg("result"), py::arg("cubes_map"),
          "2*AUC(cumN vs time fraction)-1 for n in 09:30–15:59; NaN if no trades.");
}

