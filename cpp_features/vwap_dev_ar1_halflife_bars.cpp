#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_vwap_dev_ar1_halflife_bars(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                         const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_vwap"))
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_vwap'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto vwap_arr =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_vwap"]);
    auto pbuf=price_arr.request(); auto wbuf=vwap_arr.request();
    if(pbuf.ndim!=3 || wbuf.ndim!=3) throw std::runtime_error("inputs must be 3D");
    if(pbuf.shape[0]!=wbuf.shape[0] || pbuf.shape[1]!=wbuf.shape[1] || pbuf.shape[2]!=wbuf.shape[2])
        throw std::runtime_error("last_price and interval_vwap must have same shape");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    const float* vwap_ptr =static_cast<float*>(wbuf.ptr); float* res_ptr=static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<double> dev; dev.reserve(d2);
            for(ssize_t t=0;t<d2;++t){
                float c=price_ptr[base+t]; float w=vwap_ptr[base+t];
                if(std::isnan(c) || std::isnan(w)) continue;
                double d=static_cast<double>(c)-static_cast<double>(w);
                if(std::isfinite(d)) dev.push_back(d);
            }
            size_t n=dev.size();
            if(n<4){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            size_t m=n-1; double xm=0.0, ym=0.0;
            for(size_t t=0;t<m;++t){ xm+=dev[t]; ym+=dev[t+1]; }
            xm/=m; ym/=m;
            double num=0.0, den=0.0;
            for(size_t t=0;t<m;++t){ double xd=dev[t]-xm; double yd=dev[t+1]-ym; num+=xd*yd; den+=xd*xd; }
            if(!(den>0.0)){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double phi=num/den;
            if(phi>0.0 && phi<1.0){ res_ptr[i*d1+j]=static_cast<float>(-std::log(2.0)/std::log(phi)); }
            else res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_vwap_dev_ar1_halflife_bars(py::module& m){
    m.def("cube2mat_vwap_dev_ar1_halflife_bars", &cube2mat_vwap_dev_ar1_halflife_bars,
          py::arg("result"), py::arg("cubes_map"),
          "AR(1) half-life of (close - vwap) deviation in bars.");
}

