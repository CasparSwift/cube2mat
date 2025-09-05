#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_relprice_log_elasticity_n(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                        const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price") || !cubes_map.contains("interval_n")) {
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_n'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto n_arr     = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_n"]);
    auto pbuf = price_arr.request();
    auto nbuf = n_arr.request();
    if (pbuf.ndim!=3 || nbuf.ndim!=3 ||
        pbuf.shape[0]!=nbuf.shape[0] || pbuf.shape[1]!=nbuf.shape[1] || pbuf.shape[2]!=nbuf.shape[2])
        throw std::runtime_error("arrays must have same 3D shape");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    const float* n_ptr    =static_cast<float*>(nbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);
    ssize_t rth_start=0,rth_end=d2-1; if(d2>=288){rth_start=114;rth_end=191;}

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            float anchor = price_ptr[base + rth_start];
            if (!(anchor>0.f)) { res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double sx=0, sy=0, sxx=0, sxy=0; int cnt=0;
            for(ssize_t t=rth_start; t<=rth_end; ++t){
                float c=price_ptr[base+t];
                float n=n_ptr[base+t];
                if(std::isnan(c) || std::isnan(n) || !(c>0.f) || !(n>0.f)) continue;
                double x=std::log(n);
                double y=std::log(c/anchor);
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

void bind_cube2mat_relprice_log_elasticity_n(py::module& m){
    m.def("cube2mat_relprice_log_elasticity_n", &cube2mat_relprice_log_elasticity_n,
          py::arg("result"), py::arg("cubes_map"),
          "Elasticity: slope of log(close/anchor) on log(n) within 09:30–15:59.");
}

