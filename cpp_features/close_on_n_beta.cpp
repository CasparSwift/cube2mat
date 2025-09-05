#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_close_on_n_beta(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                              const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price") || !cubes_map.contains("interval_n")) {
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_n'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto n_arr     = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_n"]);
    auto pbuf = price_arr.request();
    auto nbuf = n_arr.request();
    if (pbuf.ndim != 3 || nbuf.ndim != 3 || pbuf.shape[0]!=nbuf.shape[0] || pbuf.shape[1]!=nbuf.shape[1] || pbuf.shape[2]!=nbuf.shape[2]) {
        throw std::runtime_error("arrays must have same 3D shape");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    const float* n_ptr = static_cast<float*>(nbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i = 0; i < d0; ++i) {
        for (ssize_t j = 0; j < d1; ++j) {
            const ssize_t base = i * (d1 * d2) + j * d2;
            std::vector<double> x, y;
            x.reserve(d2); y.reserve(d2);
            for (ssize_t t = 0; t < d2; ++t) {
                float c = price_ptr[base + t];
                float n = n_ptr[base + t];
                if (std::isnan(c) || std::isnan(n)) continue;
                x.push_back(static_cast<double>(n));
                y.push_back(static_cast<double>(c));
            }
            const size_t m = x.size();
            if (m < 2) {
                res_ptr[i * d1 + j] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double xm=0.0, ym=0.0;
            for(size_t t=0;t<m;++t){xm+=x[t];ym+=y[t];}
            xm/=m; ym/=m;
            double sxx=0.0,sxy=0.0;
            for(size_t t=0;t<m;++t){double xd=x[t]-xm;double yd=y[t]-ym; sxx+=xd*xd; sxy+=xd*yd;}
            res_ptr[i*d1+j]=(sxx>0.0)?static_cast<float>(sxy/sxx):std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_close_on_n_beta(py::module& m) {
    m.def("cube2mat_close_on_n_beta", &cube2mat_close_on_n_beta,
          py::arg("result"), py::arg("cubes_map"),
          "OLS slope of close on number of trades n within 09:30–15:59.");
}

