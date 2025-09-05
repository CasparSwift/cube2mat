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

void cube2mat_shock_calm_time_median_q50(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<double> ar; ar.reserve(d2);
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                if(std::isnan(p0)||std::isnan(p1)||!(p0>0.f)||!(p1>0.f)) continue;
                double r=std::log(static_cast<double>(p1)/p0); if(!std::isfinite(r)) continue;
                ar.push_back(std::fabs(r));
            }
            size_t n=ar.size();
            if(n<2){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<double> sorted=ar; std::sort(sorted.begin(), sorted.end());
            double median=sorted[n/2];
            size_t idx=static_cast<size_t>(std::floor(0.9*n)); if(idx>=n) idx=n-1; double q90=sorted[idx];
            std::vector<double> durations; durations.reserve(n);
            for(size_t t=0;t<n;++t){
                if(ar[t]>=q90){
                    size_t k=1; while(t+k<n && ar[t+k]>median) ++k;
                    if(t+k<n) durations.push_back(static_cast<double>(k));
                }
            }
            if(durations.empty()){
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            }else{
                std::sort(durations.begin(), durations.end());
                res_ptr[i*d1+j]=static_cast<float>(durations[durations.size()/2]);
            }
        }
    }
}

void bind_cube2mat_shock_calm_time_median_q50(py::module& m){
    m.def("cube2mat_shock_calm_time_median_q50", &cube2mat_shock_calm_time_median_q50,
          py::arg("result"), py::arg("cubes_map"),
          "Median bars needed after a |r|\u2265Q90 event for |r| to drop to \u2264 median(|r|) again.");
}

