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

static int signf(float x){ return (x>0.f) - (x<0.f); }

void cube2mat_shock_nextbar_reversion_ratio(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                            const py::dict& cubes_map){
    if(!cubes_map.contains("last_price")) throw std::runtime_error("cubes_map must contain 'last_price'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf=price_arr.request();
    if(pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2||rbuf.shape[0]!=d0||rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<float> lr; lr.reserve(d2);
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1]; float p1=price_ptr[base+t];
                if(!(p0>0.f)||!(p1>0.f)||std::isnan(p0)||std::isnan(p1)) continue;
                float r=std::log(p1)-std::log(p0); if(std::isfinite(r)) lr.push_back(r);
            }
            size_t n=lr.size();
            if(n<2){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<float> absr=lr; for(float& x:absr) x=std::fabs(x); std::sort(absr.begin(), absr.end());
            float thresh=absr[static_cast<size_t>(0.90*(n-1))];
            double sum=0.0; size_t cnt=0;
            for(size_t t=0;t+1<n;++t){
                float r=lr[t]; if(std::fabs(r)<thresh) continue;
                float next=lr[t+1]; if(!std::isfinite(next)) continue;
                float val = -signf(r)*next/std::fabs(r);
                sum += val; ++cnt;
            }
            if(cnt==0){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            res_ptr[i*d1+j]=static_cast<float>(sum/cnt);
        }
    }
}

void bind_cube2mat_shock_nextbar_reversion_ratio(py::module& m){
    m.def("cube2mat_shock_nextbar_reversion_ratio", &cube2mat_shock_nextbar_reversion_ratio,
          py::arg("result"), py::arg("cubes_map"),
          "Mean of [-sign(r_t)*r_{t+1}/|r_t|] over |logret|≥Q90 events.");
}

