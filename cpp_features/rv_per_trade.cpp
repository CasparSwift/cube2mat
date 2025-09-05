#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_rv_per_trade(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                           const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("n"))
        throw std::runtime_error("cubes_map must contain 'last_price' and 'n'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto n_arr    =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["n"]);
    auto cbuf=price_arr.request(); auto nbuf=n_arr.request();
    if(cbuf.ndim!=3||nbuf.ndim!=3) throw std::runtime_error("arrays must be 3D");
    if(cbuf.shape[0]!=nbuf.shape[0]||cbuf.shape[1]!=nbuf.shape[1]||cbuf.shape[2]!=nbuf.shape[2])
        throw std::runtime_error("shape mismatch");
    const ssize_t d0=cbuf.shape[0], d1=cbuf.shape[1], d2=cbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2||rbuf.shape[0]!=d0||rbuf.shape[1]!=d1) throw std::runtime_error("result mismatch");
    const float* price_ptr=static_cast<float*>(cbuf.ptr);
    const float* n_ptr=static_cast<float*>(nbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);
#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            ssize_t base=i*(d1*d2)+j*d2;
            float sumsq=0.0f; float sumn=0.0f; float prev=std::numeric_limits<float>::quiet_NaN(); bool have=false;
            for(ssize_t t=0;t<d2;++t){
                float c=price_ptr[base+t];
                float n=n_ptr[base+t];
                if(!std::isfinite(c)||!std::isfinite(n)) continue;
                if(have){
                    float r=std::log(c)-std::log(prev);
                    sumsq+=r*r;
                }
                sumn+=n; have=true; prev=c;
            }
            res_ptr[i*d1+j]=(sumn>0.0f && have)?(sumsq/sumn):std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_rv_per_trade(py::module& m){
    m.def("cube2mat_rv_per_trade", &cube2mat_rv_per_trade,
          py::arg("result"), py::arg("cubes_map"),
          "Realized variance divided by total trade count");
}

