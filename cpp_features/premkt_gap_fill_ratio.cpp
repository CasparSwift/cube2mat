#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_premkt_gap_fill_ratio(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                    const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) throw std::runtime_error("cubes_map must contain 'last_price'");
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    const ssize_t RTH_START=114; const ssize_t RTH_END=191;

    #pragma omp parallel for collapse(2)
    for (ssize_t i=0;i<d0;++i){
        for (ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            if (d2 <= RTH_END){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            float pre_close = price_ptr[base + RTH_START - 1];
            float open_first = price_ptr[base + RTH_START];
            float close_last = price_ptr[base + RTH_END];
            if (std::isnan(pre_close) || std::isnan(open_first) || std::isnan(close_last)) {
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            float denom = open_first - pre_close;
            if (denom==0.0f) {
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            } else {
                res_ptr[i*d1+j] = - (close_last - open_first) / denom;
            }
        }
    }
}

void bind_cube2mat_premkt_gap_fill_ratio(py::module& m){
    m.def("cube2mat_premkt_gap_fill_ratio", &cube2mat_premkt_gap_fill_ratio,
          py::arg("result"), py::arg("cubes_map"),
          "Gap-fill ratio vs premarket: -(close_RTH_last - open_RTH_first) / (open_RTH_first - premarket_last_close).");
}

