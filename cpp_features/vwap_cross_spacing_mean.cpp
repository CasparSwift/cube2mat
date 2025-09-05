#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_vwap_cross_spacing_mean(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                      const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_vwap"))
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_vwap'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto vwap_arr =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_vwap"]);
    auto pbuf=price_arr.request(); auto vbuf=vwap_arr.request();
    if(pbuf.ndim!=3 || vbuf.ndim!=3) throw std::runtime_error("arrays must be 3D");
    if(pbuf.shape[0]!=vbuf.shape[0]||pbuf.shape[1]!=vbuf.shape[1]||pbuf.shape[2]!=vbuf.shape[2])
        throw std::runtime_error("shape mismatch");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1)
        throw std::runtime_error("result must be 2D (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    const float* vwap_ptr =static_cast<float*>(vbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);
    const double TOT_MIN=389.0; // session minutes

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<int> idx; idx.reserve(d2);
            std::vector<int> sign; sign.reserve(d2);
            for(ssize_t t=0;t<d2;++t){
                float c=price_ptr[base+t];
                float v=vwap_ptr[base+t];
                if(std::isnan(c) || std::isnan(v)) continue;
                float diff=c-v;
                if(diff>0) {sign.push_back(1); idx.push_back(t);} 
                else if(diff<0){sign.push_back(-1); idx.push_back(t);} // zero diff dropped
            }
            size_t m=sign.size();
            if(m<2){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<int> crosses;
            for(size_t t=1;t<m;++t){ if(sign[t]!=sign[t-1]) crosses.push_back(idx[t]); }
            if(crosses.size()<2){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double sum_gap=0.0; size_t cnt=0;
            for(size_t t=1;t<crosses.size();++t){ sum_gap += (crosses[t]-crosses[t-1])*5.0; cnt++; }
            if(cnt==0){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            res_ptr[i*d1+j]=static_cast<float>((sum_gap/cnt)/TOT_MIN);
        }
    }
}

void bind_cube2mat_vwap_cross_spacing_mean(py::module& m){
    m.def("cube2mat_vwap_cross_spacing_mean", &cube2mat_vwap_cross_spacing_mean,
          py::arg("result"), py::arg("cubes_map"),
          "Mean spacing between VWAP crossings as fraction of 389 minutes; NaN if <2 crossings.");
}

