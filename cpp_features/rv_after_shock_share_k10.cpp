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

void cube2mat_rv_after_shock_share_k10(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                       const py::dict& cubes_map){
    if(!cubes_map.contains("last_price"))
        throw std::runtime_error("cubes_map must contain 'last_price'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf=price_arr.request();
    if(pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2||rbuf.shape[0]!=d0||rbuf.shape[1]!=d1)
        throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<float> rets; rets.reserve(d2);
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                if(!(p0>0.f)||!(p1>0.f)||std::isnan(p0)||std::isnan(p1)) continue;
                float r=(p1-p0)/p0;
                if(std::isfinite(r)) rets.push_back(r);
            }
            size_t n=rets.size();
            if(n==0){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double total_rv=0.0; for(float r:rets) total_rv+= (double)r*r;
            if(!(total_rv>0.0)){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<float> absr=rets; for(float& x:absr) x=std::fabs(x); std::sort(absr.begin(), absr.end());
            float thresh=absr[static_cast<size_t>(0.90*(n-1))];
            std::vector<bool> flag(n,false);
            for(size_t t=0;t<n;++t){
                if(std::fabs(rets[t])>=thresh){
                    for(size_t k=t+1;k<=t+10 && k<n;++k) flag[k]=true;
                }
            }
            double rv_shock=0.0; for(size_t t=0;t<n;++t) if(flag[t]) rv_shock+= (double)rets[t]*rets[t];
            res_ptr[i*d1+j]=static_cast<float>(rv_shock/total_rv);
        }
    }
}

void bind_cube2mat_rv_after_shock_share_k10(py::module& m){
    m.def("cube2mat_rv_after_shock_share_k10", &cube2mat_rv_after_shock_share_k10,
          py::arg("result"), py::arg("cubes_map"),
          "Share of RV in 10-bar windows after |r|≥Q90 shocks.");
}

