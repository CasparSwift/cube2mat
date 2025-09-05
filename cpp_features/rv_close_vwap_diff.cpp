#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_rv_close_vwap_diff(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                 const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price") || !cubes_map.contains("interval_vwap")) {
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_vwap'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto vwap_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_vwap"]);
    auto pbuf = price_arr.request();
    auto wbuf = vwap_arr.request();
    if (pbuf.ndim != 3 || wbuf.ndim != 3 ||
        pbuf.shape[0]!=wbuf.shape[0] || pbuf.shape[1]!=wbuf.shape[1] || pbuf.shape[2]!=wbuf.shape[2]) {
        throw std::runtime_error("last_price and interval_vwap must have same 3D shape");
    }
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim !=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    const float* vwap_ptr  = static_cast<float*>(wbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            double sum=0.0; int cnt=0;
            float prev_p=std::numeric_limits<float>::quiet_NaN();
            float prev_w=std::numeric_limits<float>::quiet_NaN();
            for(ssize_t t=0;t<d2;++t){
                float p=price_ptr[base+t];
                float w=vwap_ptr[base+t];
                if(std::isnan(p)||std::isnan(w)) { prev_p=std::numeric_limits<float>::quiet_NaN(); prev_w=std::numeric_limits<float>::quiet_NaN(); continue; }
                if(t>0 && !std::isnan(prev_p) && !std::isnan(prev_w)){
                    double diff = (static_cast<double>(p)-w) - (static_cast<double>(prev_p)-prev_w);
                    sum += diff*diff;
                    ++cnt;
                }
                prev_p=p; prev_w=w;
            }
            res_ptr[i*d1+j] = (cnt>0)? static_cast<float>(sum): std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_rv_close_vwap_diff(py::module& m){
    m.def("cube2mat_rv_close_vwap_diff", &cube2mat_rv_close_vwap_diff,
          py::arg("result"), py::arg("cubes_map"),
          "Realized variance of (close - vwap): sum of squared first differences.");
}

