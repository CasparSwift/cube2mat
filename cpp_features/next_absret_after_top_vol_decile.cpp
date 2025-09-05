#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_next_absret_after_top_vol_decile(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                               const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_volume"))
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_volume'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto vol_arr  =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_volume"]);
    auto pbuf=price_arr.request(); auto vbuf=vol_arr.request();
    if(pbuf.ndim!=3 || vbuf.ndim!=3) throw std::runtime_error("arrays must be 3D");
    if(pbuf.shape[0]!=vbuf.shape[0] || pbuf.shape[1]!=vbuf.shape[1] || pbuf.shape[2]!=vbuf.shape[2])
        throw std::runtime_error("last_price and interval_volume shape mismatch");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result mismatch");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    const float* vol_ptr  =static_cast<float*>(vbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            ssize_t base=i*(d1*d2)+j*d2;
            std::vector<float> vols; vols.reserve(d2);
            for(ssize_t t=0;t<d2;++t){
                float v=vol_ptr[base+t];
                if(std::isfinite(v)) vols.push_back(v);
            }
            if(vols.size()<2){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::sort(vols.begin(), vols.end());
            float thr = vols[static_cast<size_t>(std::floor(0.9*(vols.size()-1)))];
            float sumabs=0.0f; int cnt=0;
            float prev=std::numeric_limits<float>::quiet_NaN(); bool prev_valid=false;
            for(ssize_t t=0;t<d2-1;++t){
                float v=vol_ptr[base+t];
                float p=price_ptr[base+t];
                float nextp=price_ptr[base+t+1];
                if(!std::isfinite(v) || v<thr || !std::isfinite(p) || !std::isfinite(nextp)) continue;
                float r=std::log(nextp)-std::log(p);
                if(std::isfinite(r)){ sumabs+=std::fabs(r); cnt++; }
            }
            res_ptr[i*d1+j]=(cnt>0)?(sumabs/cnt):std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_next_absret_after_top_vol_decile(py::module& m){
    m.def("cube2mat_next_absret_after_top_vol_decile", &cube2mat_next_absret_after_top_vol_decile,
          py::arg("result"), py::arg("cubes_map"),
          "E[|next log return| | current volume in top decile]");
}

