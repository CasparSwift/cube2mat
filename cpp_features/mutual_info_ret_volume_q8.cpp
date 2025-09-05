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

void cube2mat_mutual_info_ret_volume_q8(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                        const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_volume"))
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_volume'");
    auto price_arr = py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto vol_arr   = py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_volume"]);
    auto pbuf = price_arr.request(); auto vbuf = vol_arr.request();
    if(pbuf.ndim!=3 || vbuf.ndim!=3) throw std::runtime_error("arrays must be 3D");
    if(pbuf.shape[0]!=vbuf.shape[0] || pbuf.shape[1]!=vbuf.shape[1] || pbuf.shape[2]!=vbuf.shape[2])
        throw std::runtime_error("last_price and interval_volume shape mismatch");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result mismatch");
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    const float* vol_ptr   = static_cast<float*>(vbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            ssize_t base=i*(d1*d2)+j*d2;
            std::vector<float> rvec; rvec.reserve(d2);
            std::vector<float> vvec; vvec.reserve(d2);
            float prev=std::numeric_limits<float>::quiet_NaN();
            bool prev_valid=false;
            for(ssize_t t=0;t<d2;++t){
                float p=price_ptr[base+t];
                float v=vol_ptr[base+t];
                if(!std::isfinite(v)) continue;
                if(prev_valid && std::isfinite(p)){
                    float r=std::log(p)-std::log(prev);
                    if(std::isfinite(r)){
                        rvec.push_back(r);
                        vvec.push_back(v);
                    }
                }
                if(std::isfinite(p)) { prev=p; prev_valid=true; }
            }
            size_t n=rvec.size();
            if(n<8){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<float> rsorted=rvec, vsorted=vvec;
            std::sort(rsorted.begin(), rsorted.end());
            std::sort(vsorted.begin(), vsorted.end());
            float rq[7], vq[7];
            for(int k=1;k<8;++k){
                rq[k-1]=rsorted[static_cast<size_t>(std::floor((n*k)/8.0)) - 1];
                vq[k-1]=vsorted[static_cast<size_t>(std::floor((n*k)/8.0)) - 1];
            }
            int counts[8][8]={0};
            for(size_t idx=0; idx<n; ++idx){
                float r=rvec[idx]; float v=vvec[idx];
                int rb=std::lower_bound(rq, rq+7, r)-rq;
                int vb=std::lower_bound(vq, vq+7, v)-vq;
                counts[rb][vb]++;
            }
            std::vector<float> pr(8,0.0f), pv(8,0.0f);
            for(int a=0;a<8;++a){
                for(int b=0;b<8;++b){
                    float pij=counts[a][b]/static_cast<float>(n);
                    pr[a]+=pij;
                    pv[b]+=pij;
                }
            }
            double mi=0.0;
            for(int a=0;a<8;++a){
                for(int b=0;b<8;++b){
                    int c=counts[a][b];
                    if(c==0) continue;
                    double pij=c/static_cast<double>(n);
                    mi += pij * std::log(pij/(pr[a]*pv[b]));
                }
            }
            res_ptr[i*d1+j]=static_cast<float>(mi / std::log(2.0));
        }
    }
}

void bind_cube2mat_mutual_info_ret_volume_q8(py::module& m){
    m.def("cube2mat_mutual_info_ret_volume_q8", &cube2mat_mutual_info_ret_volume_q8,
          py::arg("result"), py::arg("cubes_map"),
          "Mutual information between log returns and interval volume using 8x8 quantile bins");
}

