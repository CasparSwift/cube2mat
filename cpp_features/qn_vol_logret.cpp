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

static float qn_scale(const std::vector<float>& arr){
    size_t n=arr.size();
    if(n<3) return std::numeric_limits<float>::quiet_NaN();
    std::vector<float> diffs;
    diffs.reserve(n*(n-1)/2);
    for(size_t i=0;i<n-1;++i){
        for(size_t j=i+1;j<n;++j){
            diffs.push_back(std::fabs(arr[j]-arr[i]));
        }
    }
    size_t m=diffs.size();
    if(m==0) return std::numeric_limits<float>::quiet_NaN();
    std::sort(diffs.begin(), diffs.end());
    size_t idx = static_cast<size_t>(std::floor(0.25 * (m - 1)));
    float q1 = diffs[idx];
    if(!std::isfinite(q1)) return std::numeric_limits<float>::quiet_NaN();
    return static_cast<float>(2.2219 * q1);
}

void cube2mat_qn_vol_logret(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<float> r; r.reserve(d2);
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                if(!(p0>0.f) || !(p1>0.f) || std::isnan(p0) || std::isnan(p1)) continue;
                r.push_back(std::log(p1/p0));
            }
            res_ptr[i*d1+j]=qn_scale(r);
        }
    }
}

void bind_cube2mat_qn_vol_logret(py::module& m){
    m.def("cube2mat_qn_vol_logret", &cube2mat_qn_vol_logret,
          py::arg("result"), py::arg("cubes_map"),
          "Robust Qn scale (2.2219 * 1st quartile of pairwise |Δr|) for log returns.");
}

