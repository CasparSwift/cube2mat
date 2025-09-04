#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_corr_ret_absnextret(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                  const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) {
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim != 3) {
        throw std::runtime_error("last_price must be 3D array");
    }
    const ssize_t d0 = pbuf.shape[0];
    const ssize_t d1 = pbuf.shape[1];
    const ssize_t d2 = pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim != 2 || rbuf.shape[0] != d0 || rbuf.shape[1] != d1) {
        throw std::runtime_error("result must be 2D with shape (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for (ssize_t i=0;i<d0;++i){
        for (ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            double sx=0.0, sy=0.0, sxx=0.0, syy=0.0, sxy=0.0; int cnt=0;
            for(ssize_t t=1;t<d2-1;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                float p2=price_ptr[base+t+1];
                if(std::isnan(p0)||std::isnan(p1)||std::isnan(p2)) continue;
                float r  = std::log(p1)-std::log(p0);
                float rn = std::log(p2)-std::log(p1);
                if(std::isnan(r)||std::isnan(rn)) continue;
                float x=r;
                float y=std::fabs(rn);
                sx+=x; sy+=y; sxx+=x*x; syy+=y*y; sxy+=x*y; ++cnt;
            }
            if(cnt>1){
                double cov = sxy - sx*sy/cnt;
                double varx = sxx - sx*sx/cnt;
                double vary = syy - sy*sy/cnt;
                res_ptr[i*d1+j]=(varx>0.0 && vary>0.0)?static_cast<float>(cov/std::sqrt(varx*vary))
                                                      : std::numeric_limits<float>::quiet_NaN();
            }else{
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_corr_ret_absnextret(py::module& m){
    m.def("cube2mat_corr_ret_absnextret", &cube2mat_corr_ret_absnextret,
          py::arg("result"), py::arg("cubes_map"),
          "Correlation between ret_t and |ret_{t+1}| as leverage-effect proxy.");
}

