#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_absret_on_sqrtn_beta(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                   const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_n")){
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_n'");
    }
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto n_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_n"]);
    auto pbuf=price_arr.request();
    auto nbuf=n_arr.request();
    if(pbuf.ndim!=3||nbuf.ndim!=3) throw std::runtime_error("inputs must be 3D");
    if(pbuf.shape[0]!=nbuf.shape[0]||pbuf.shape[1]!=nbuf.shape[1]||pbuf.shape[2]!=nbuf.shape[2])
        throw std::runtime_error("arrays must have same shape");
    const ssize_t d0=pbuf.shape[0];
    const ssize_t d1=pbuf.shape[1];
    const ssize_t d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2||rbuf.shape[0]!=d0||rbuf.shape[1]!=d1) throw std::runtime_error("result must be 2D");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    const float* n_ptr=static_cast<float*>(nbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            double sx=0.0, sy=0.0, sxx=0.0, sxy=0.0; int cnt=0;
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                float n=n_ptr[base+t];
                if(std::isnan(p0)||std::isnan(p1)||std::isnan(n)||!(n>0.f)) continue;
                float r=std::log(p1)-std::log(p0);
                if(std::isnan(r)) continue;
                float x=std::sqrt(n);
                float y=std::fabs(r);
                sx+=x; sy+=y; sxx+=x*x; sxy+=x*y; ++cnt;
            }
            if(cnt>1){
                double cov = sxy - sx*sy/cnt;
                double varx = sxx - sx*sx/cnt;
                res_ptr[i*d1+j]=(varx>0.0)?static_cast<float>(cov/varx)
                                            : std::numeric_limits<float>::quiet_NaN();
            }else{
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_absret_on_sqrtn_beta(py::module& m){
    m.def("cube2mat_absret_on_sqrtn_beta", &cube2mat_absret_on_sqrtn_beta,
          py::arg("result"), py::arg("cubes_map"),
          "Elasticity (slope) of |logret| on sqrt(n) per bar.");
}

