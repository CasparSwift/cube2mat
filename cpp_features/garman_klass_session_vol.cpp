#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_garman_klass_session_vol(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                       const py::dict& cubes_map) {
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_high") || !cubes_map.contains("interval_low")){
        throw std::runtime_error("cubes_map must contain 'last_price','interval_high','interval_low'");
    }
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto high_arr =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_high"]);
    auto low_arr  =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_low"]);
    auto pbuf=price_arr.request(); auto hbuf=high_arr.request(); auto lbuf=low_arr.request();
    if(pbuf.ndim!=3 || hbuf.ndim!=3 || lbuf.ndim!=3 ||
       pbuf.shape[0]!=hbuf.shape[0] || pbuf.shape[1]!=hbuf.shape[1] || pbuf.shape[2]!=hbuf.shape[2] ||
       pbuf.shape[0]!=lbuf.shape[0] || pbuf.shape[1]!=lbuf.shape[1] || pbuf.shape[2]!=lbuf.shape[2])
        throw std::runtime_error("arrays must have same 3D shape");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    const float* high_ptr =static_cast<float*>(hbuf.ptr);
    const float* low_ptr  =static_cast<float*>(lbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);
    const double k = 2.0*std::log(2.0)-1.0;

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            float O=price_ptr[base];
            float C=price_ptr[base+d2-1];
            float H=-std::numeric_limits<float>::infinity();
            float L= std::numeric_limits<float>::infinity();
            for(ssize_t t=0;t<d2;++t){
                float hh=high_ptr[base+t];
                float ll=low_ptr[base+t];
                if(!std::isnan(hh)) H=std::max(H,hh);
                if(!std::isnan(ll)) L=std::min(L,ll);
            }
            if(!(O>0.f) || !(C>0.f) || !(H>0.f) || !(L>0.f) || !(H>L)){
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double lnHL=std::log(H/L);
            double lnCO=std::log(C/O);
            double var_gk = 0.5*lnHL*lnHL - k*lnCO*lnCO;
            if(!std::isfinite(var_gk) || var_gk<0.0){
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            }else{
                res_ptr[i*d1+j]=static_cast<float>(std::sqrt(var_gk));
            }
        }
    }
}

void bind_cube2mat_garman_klass_session_vol(py::module& m){
    m.def("cube2mat_garman_klass_session_vol", &cube2mat_garman_klass_session_vol,
          py::arg("result"), py::arg("cubes_map"),
          "Session-level Garman–Klass volatility using aggregated O/H/L/C.");
}
