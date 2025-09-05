#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_rv_logret_15m(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                             const py::dict& cubes_map){
    if(!cubes_map.contains("last_price")) throw std::runtime_error("cubes_map must contain 'last_price'");
    auto price_arr=py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf=price_arr.request();
    if(pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

    const double TOTAL_MIN=389.0;
    const double interval_min=(d2>1)? TOTAL_MIN/static_cast<double>(d2-1) : TOTAL_MIN;
    ssize_t step=static_cast<ssize_t>(std::llround(15.0/interval_min));
    if(step<1) step=1;

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            double sum=0.0; int cnt=0;
            for(ssize_t t=step;t<d2;t+=step){
                float p0=price_ptr[base+t-step];
                float p1=price_ptr[base+t];
                if(std::isnan(p0)||std::isnan(p1)||!(p0>0.f)||!(p1>0.f)) continue;
                double r=std::log(static_cast<double>(p1)/p0); if(!std::isfinite(r)) continue;
                sum+=r*r; ++cnt;
            }
            res_ptr[i*d1+j]=(cnt>0)?static_cast<float>(sum):std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_rv_logret_15m(py::module& m){
    m.def("cube2mat_rv_logret_15m", &cube2mat_rv_logret_15m,
          py::arg("result"), py::arg("cubes_map"),
          "Realized variance of log returns on 15-minute resampled close within 09:30–15:59.");
}

