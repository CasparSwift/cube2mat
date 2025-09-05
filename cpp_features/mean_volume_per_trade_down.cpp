#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_mean_volume_per_trade_down(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                         const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_volume") || !cubes_map.contains("n"))
        throw std::runtime_error("cubes_map must contain 'last_price','interval_volume','n'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto vol_arr  =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_volume"]);
    auto n_arr    =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["n"]);
    auto cbuf=price_arr.request(); auto vbuf=vol_arr.request(); auto nbuf=n_arr.request();
    if(cbuf.ndim!=3||vbuf.ndim!=3||nbuf.ndim!=3) throw std::runtime_error("arrays must be 3D");
    if(cbuf.shape[0]!=vbuf.shape[0]||cbuf.shape[1]!=vbuf.shape[1]||cbuf.shape[2]!=vbuf.shape[2]||
       cbuf.shape[0]!=nbuf.shape[0]||cbuf.shape[1]!=nbuf.shape[1]||cbuf.shape[2]!=nbuf.shape[2])
        throw std::runtime_error("shape mismatch");
    const ssize_t d0=cbuf.shape[0], d1=cbuf.shape[1], d2=cbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2||rbuf.shape[0]!=d0||rbuf.shape[1]!=d1) throw std::runtime_error("result mismatch");
    const float* price_ptr=static_cast<float*>(cbuf.ptr);
    const float* vol_ptr=static_cast<float*>(vbuf.ptr);
    const float* n_ptr=static_cast<float*>(nbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);
#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            ssize_t base=i*(d1*d2)+j*d2;
            float sum=0.0f; int cnt=0;
            for(ssize_t t=1;t<d2;++t){
                float c0=price_ptr[base+t-1];
                float c1=price_ptr[base+t];
                float v=vol_ptr[base+t];
                float n=n_ptr[base+t];
                if(!std::isfinite(c0)||!std::isfinite(c1)||!std::isfinite(v)||!std::isfinite(n)||n<=0) continue;
                if(c1<c0){
                    sum+=v/n; cnt++;
                }
            }
            res_ptr[i*d1+j]=(cnt>0)?(sum/cnt):std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_mean_volume_per_trade_down(py::module& m){
    m.def("cube2mat_mean_volume_per_trade_down", &cube2mat_mean_volume_per_trade_down,
          py::arg("result"), py::arg("cubes_map"),
          "Average interval_volume per trade on down bars");
}

