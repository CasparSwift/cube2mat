#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_volume_weighted_simple_return(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                            const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_volume"))
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_volume'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto vol_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_volume"]);
    auto pbuf=price_arr.request(); auto vbuf=vol_arr.request();
    if(pbuf.ndim!=3 || vbuf.ndim!=3) throw std::runtime_error("inputs must be 3D");
    if(pbuf.shape[0]!=vbuf.shape[0] || pbuf.shape[1]!=vbuf.shape[1] || pbuf.shape[2]!=vbuf.shape[2])
        throw std::runtime_error("last_price and interval_volume must have same shape");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    const float* vol_ptr=static_cast<float*>(vbuf.ptr); float* res_ptr=static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            double num=0.0, den=0.0;
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                float v=vol_ptr[base+t];
                if(!(p0>0.f) || !(p1>0.f) || std::isnan(p0) || std::isnan(p1) || std::isnan(v) || !(v>0.f)) continue;
                double r=p1/p0 - 1.0;
                if(!std::isfinite(r)) continue;
                num += v*r;
                den += v;
            }
            res_ptr[i*d1+j] = (den>0.0)? static_cast<float>(num/den)
                                        : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_volume_weighted_simple_return(py::module& m){
    m.def("cube2mat_volume_weighted_simple_return", &cube2mat_volume_weighted_simple_return,
          py::arg("result"), py::arg("cubes_map"),
          "Volume-weighted mean of simple returns within RTH.");
}

