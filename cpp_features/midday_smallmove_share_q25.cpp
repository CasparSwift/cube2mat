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

void cube2mat_midday_smallmove_share_q25(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                         const py::dict& cubes_map){
    if(!cubes_map.contains("last_price"))
        throw std::runtime_error("cubes_map must contain 'last_price'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf=price_arr.request();
    if(pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1)
        throw std::runtime_error("result must be 2D (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

    const size_t mid_start = 17; // return index corresponding to 11:00
    const size_t mid_end   = 53; // 14:00

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<float> absrets; absrets.reserve(d2);
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                if(!(p0>0.f) || !(p1>0.f) || std::isnan(p0) || std::isnan(p1)) continue;
                float r=std::log(p1/p0);
                absrets.push_back(std::fabs(r));
            }
            size_t n=absrets.size();
            if(n==0){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<float> sorted=absrets; std::sort(sorted.begin(), sorted.end());
            size_t idx = static_cast<size_t>(std::floor(0.25 * (sorted.size()-1)));
            float q25 = sorted[idx];
            size_t count=0; size_t tot=0;
            for(size_t t=mid_start; t<=mid_end && t<n; ++t){
                float ar=absrets[t];
                if(std::isnan(ar)) continue;
                tot++; if(ar<=q25) count++;
            }
            res_ptr[i*d1+j]= (tot>0)? static_cast<float>(count/static_cast<double>(tot)) : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_midday_smallmove_share_q25(py::module& m){
    m.def("cube2mat_midday_smallmove_share_q25", &cube2mat_midday_smallmove_share_q25,
          py::arg("result"), py::arg("cubes_map"),
          "Share of bars in 11:00–14:00 with |logret| ≤ overall RTH 25th percentile of |logret|.");
}

