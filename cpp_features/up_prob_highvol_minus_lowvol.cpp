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

static float quantile_ignore_nan(std::vector<float> v, double q){
    size_t n=v.size();
    if(n==0) return std::numeric_limits<float>::quiet_NaN();
    double pos=q*(n-1); size_t idx=static_cast<size_t>(pos);
    std::nth_element(v.begin(), v.begin()+idx, v.end());
    float val=v[idx]; size_t idx2=idx+1;
    if(idx2<n){ std::nth_element(v.begin(), v.begin()+idx2, v.end()); float val2=v[idx2]; val = static_cast<float>(val + (pos-idx)*(val2-val)); }
    return val;
}

void cube2mat_up_prob_highvol_minus_lowvol(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                           const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_volume"))
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_volume'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto vol_arr  =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_volume"]);
    auto pbuf=price_arr.request(); auto vbuf=vol_arr.request();
    if(pbuf.ndim!=3 || vbuf.ndim!=3 || pbuf.shape[0]!=vbuf.shape[0] || pbuf.shape[1]!=vbuf.shape[1] || pbuf.shape[2]!=vbuf.shape[2])
        throw std::runtime_error("arrays must have same shape");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr); const float* vol_ptr=static_cast<float*>(vbuf.ptr); float* res_ptr=static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<double> rets; rets.reserve(d2);
            std::vector<float> vols; vols.reserve(d2);
            float prev=std::numeric_limits<float>::quiet_NaN();
            for(ssize_t t=0;t<d2;++t){
                float c=price_ptr[base+t]; float v=vol_ptr[base+t];
                if(std::isnan(c) || !(c>0.f)){ prev=std::numeric_limits<float>::quiet_NaN(); continue; }
                if(!std::isnan(prev) && prev>0.f){
                    rets.push_back(std::log(c/prev));
                    vols.push_back(v); // may be NaN
                }
                prev=c;
            }
            size_t n=rets.size();
            if(n==0){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<float> vvalid; vvalid.reserve(n);
            for(float v: vols) if(!std::isnan(v)) vvalid.push_back(v);
            if(vvalid.empty()){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            float v90=quantile_ignore_nan(vvalid,0.90); float v10=quantile_ignore_nan(vvalid,0.10);
            if(std::isnan(v90) || std::isnan(v10)){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double ph_sum=0.0, pl_sum=0.0; size_t hv_cnt=0, lv_cnt=0;
            for(size_t k=0;k<n;++k){ float v=vols[k]; double r=rets[k];
                if(!std::isnan(v)){
                    if(v>=v90){ hv_cnt++; if(r>0) ph_sum+=1.0; }
                    if(v<=v10){ lv_cnt++; if(r>0) pl_sum+=1.0; }
                }
            }
            if(hv_cnt==0 || lv_cnt==0){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double ph=ph_sum/hv_cnt; double pl=pl_sum/lv_cnt;
            res_ptr[i*d1+j]=static_cast<float>((std::isfinite(ph)&&std::isfinite(pl))?(ph-pl):std::numeric_limits<float>::quiet_NaN());
        }
    }
}

void bind_cube2mat_up_prob_highvol_minus_lowvol(py::module& m){
    m.def("cube2mat_up_prob_highvol_minus_lowvol", &cube2mat_up_prob_highvol_minus_lowvol,
          py::arg("result"), py::arg("cubes_map"),
          "ΔP(up): P(r>0 | volume≥Q90) − P(r>0 | volume≤Q10) using RTH log returns aligned to volume at time t.");
}

