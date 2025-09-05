#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_cv_trade_size(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                            const py::dict& cubes_map){
    if(!cubes_map.contains("interval_volume") || !cubes_map.contains("n"))
        throw std::runtime_error("cubes_map must contain 'interval_volume' and 'n'");
    auto v_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_volume"]);
    auto n_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["n"]);
    auto vbuf=v_arr.request(); auto nbuf=n_arr.request();
    if(vbuf.ndim!=3||nbuf.ndim!=3) throw std::runtime_error("arrays must be 3D");
    if(vbuf.shape[0]!=nbuf.shape[0]||vbuf.shape[1]!=nbuf.shape[1]||vbuf.shape[2]!=nbuf.shape[2])
        throw std::runtime_error("shape mismatch");
    const ssize_t d0=vbuf.shape[0], d1=vbuf.shape[1], d2=vbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2||rbuf.shape[0]!=d0||rbuf.shape[1]!=d1) throw std::runtime_error("result shape mismatch");
    const float* v_ptr=static_cast<float*>(vbuf.ptr);
    const float* n_ptr=static_cast<float*>(nbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);
#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            ssize_t base=i*(d1*d2)+j*d2;
            float sum=0.0f,sumsq=0.0f;int cnt=0;
            for(ssize_t t=0;t<d2;++t){
                float v=v_ptr[base+t];
                float n=n_ptr[base+t];
                if(!std::isfinite(v)||!std::isfinite(n)||n<=0) continue;
                float ts=v/n;
                sum+=ts; sumsq+=ts*ts; cnt++;
            }
            if(cnt>=3 && sum>0){
                float mean=sum/cnt;
                float var=(sumsq - (sum*sum)/cnt)/(cnt-1);
                float sd=std::sqrt(std::max(var,0.0f));
                res_ptr[i*d1+j]=sd/mean;
            }else res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_cv_trade_size(py::module& m){
    m.def("cube2mat_cv_trade_size", &cube2mat_cv_trade_size,
          py::arg("result"), py::arg("cubes_map"),
          "Coefficient of variation of per-bar trade size");
}

