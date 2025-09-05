#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_ret_next_on_vwapdev_beta(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                       const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price") || !cubes_map.contains("interval_vwap")) {
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_vwap'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto vwap_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_vwap"]);
    auto pbuf = price_arr.request();
    auto vwbuf = vwap_arr.request();
    if (pbuf.ndim!=3 || vwbuf.ndim!=3 ||
        pbuf.shape[0]!=vwbuf.shape[0] || pbuf.shape[1]!=vwbuf.shape[1] || pbuf.shape[2]!=vwbuf.shape[2])
        throw std::runtime_error("arrays must have same 3D shape");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    const float* vwap_ptr =static_cast<float*>(vwbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);
    ssize_t rth_start=0,rth_end=d2-1; if(d2>=288){rth_start=114;rth_end=191;}

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            double sx=0, sy=0, sxx=0, sxy=0; int cnt=0;
            for(ssize_t t=rth_start; t<rth_end; ++t){
                float c=price_ptr[base+t];
                float c_next=price_ptr[base+t+1];
                float vw=vwap_ptr[base+t];
                if(std::isnan(c) || std::isnan(c_next) || std::isnan(vw) || !(c>0.f) || !(c_next>0.f) || !(vw>0.f)) continue;
                double x=(c - vw)/vw;
                double y=std::log(c_next/c);
                sx+=x; sy+=y; sxx+=x*x; sxy+=x*y; ++cnt;
            }
            if(cnt>1){
                double varx=sxx - sx*sx/cnt;
                if(varx>0){
                    double cov=sxy - sx*sy/cnt;
                    res_ptr[i*d1+j]=static_cast<float>(cov/varx);
                    continue;
                }
            }
            res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_ret_next_on_vwapdev_beta(py::module& m){
    m.def("cube2mat_ret_next_on_vwapdev_beta", &cube2mat_ret_next_on_vwapdev_beta,
          py::arg("result"), py::arg("cubes_map"),
          "Beta of next ret on (close-vwap)/vwap deviation (RTH).");
}

