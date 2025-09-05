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

static float quantile(std::vector<float>& v, double q){
    size_t n=v.size();
    if(n==0) return std::numeric_limits<float>::quiet_NaN();
    double pos = q*(n-1);
    size_t idx = static_cast<size_t>(pos);
    std::nth_element(v.begin(), v.begin()+idx, v.end());
    float val = v[idx];
    size_t idx2 = idx+1;
    if(idx2 < n){
        std::nth_element(v.begin(), v.begin()+idx2, v.end());
        float val2=v[idx2];
        val = static_cast<float>(val + (pos - idx)*(val2 - val));
    }
    return val;
}

void cube2mat_next3_absret_highvol_minus_lowvol(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                                const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_volume"))
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_volume'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto vol_arr  =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_volume"]);
    auto pbuf=price_arr.request(); auto vbuf=vol_arr.request();
    if(pbuf.ndim!=3 || vbuf.ndim!=3) throw std::runtime_error("arrays must be 3D");
    if(pbuf.shape[0]!=vbuf.shape[0]||pbuf.shape[1]!=vbuf.shape[1]||pbuf.shape[2]!=vbuf.shape[2])
        throw std::runtime_error("shape mismatch");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1)
        throw std::runtime_error("result must be 2D (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    const float* vol_ptr  =static_cast<float*>(vbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<float> r; r.reserve(d2);
            std::vector<float> v; v.reserve(d2);
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                float vol=vol_ptr[base+t];
                if(!(p0>0.f) || !(p1>0.f) || std::isnan(p0) || std::isnan(p1) || std::isnan(vol)) continue;
                r.push_back(std::log(p1/p0));
                v.push_back(vol);
            }
            size_t n=r.size();
            if(n<4){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<float> vcopy=v; float v90=quantile(vcopy,0.90); vcopy=v; float v10=quantile(vcopy,0.10);
            double hv_sum=0.0, lv_sum=0.0; size_t hv_cnt=0, lv_cnt=0;
            for(size_t t=0;t<n;++t){
                float vol=v[t];
                if(vol>=v90){
                    double s=0.0; for(size_t k=t+1;k<=std::min(t+3,n-1);++k) s+=std::fabs(r[k]);
                    hv_sum+=s; hv_cnt++; }
                if(vol<=v10){
                    double s=0.0; for(size_t k=t+1;k<=std::min(t+3,n-1);++k) s+=std::fabs(r[k]);
                    lv_sum+=s; lv_cnt++; }
            }
            if(hv_cnt==0 || lv_cnt==0){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double hvm=hv_sum/hv_cnt; double lvm=lv_sum/lv_cnt;
            if(!std::isfinite(hvm) || !std::isfinite(lvm)) res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            else res_ptr[i*d1+j]=static_cast<float>(hvm - lvm);
        }
    }
}

void bind_cube2mat_next3_absret_highvol_minus_lowvol(py::module& m){
    m.def("cube2mat_next3_absret_highvol_minus_lowvol", &cube2mat_next3_absret_highvol_minus_lowvol,
          py::arg("result"), py::arg("cubes_map"),
          "E[Σ_{i=1..3}|r_{t+i}| | vol_t≥Q90] − E[Σ_{i=1..3}|r_{t+i}| | vol_t≤Q10], within RTH.");
}

