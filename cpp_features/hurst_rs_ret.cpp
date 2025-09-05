#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static float hurst_rs(const std::vector<float>& r){
    const int n = static_cast<int>(r.size());
    const int sizes[] = {5,10,20,40,80,160};
    std::vector<std::pair<int,double>> pts;
    for(int s : sizes){
        if(s < 5 || s > n/2) continue;
        int k = n / s;
        if(k < 1) continue;
        double sum_rs = 0.0; int cnt = 0;
        for(int i=0;i<k;++i){
            double mean=0.0;
            for(int j=0;j<s;++j) mean += r[i*s+j];
            mean /= s;
            double Z=0.0, maxZ=-1e100, minZ=1e100, var=0.0;
            for(int j=0;j<s;++j){
                double x = r[i*s+j] - mean;
                Z += x;
                if(Z>maxZ) maxZ=Z; if(Z<minZ) minZ=Z;
                var += x*x;
            }
            if(s>1){
                double R = maxZ - minZ;
                double S = std::sqrt(var/(s-1));
                if(S>0.0 && std::isfinite(R) && std::isfinite(S)){
                    sum_rs += R/S; ++cnt;
                }
            }
        }
        if(cnt>0){
            pts.emplace_back(s, sum_rs/cnt);
        }
    }
    if(pts.size()<2) return std::numeric_limits<float>::quiet_NaN();
    double sumx=0.0,sumy=0.0,sumxx=0.0,sumxy=0.0; int m=pts.size();
    for(auto &p: pts){
        double lx=std::log((double)p.first);
        double ly=std::log(p.second);
        sumx+=lx; sumy+=ly; sumxx+=lx*lx; sumxy+=lx*ly;
    }
    double den = sumxx - sumx*sumx/m;
    double num = sumxy - sumx*sumy/m;
    if(den<=0.0) return std::numeric_limits<float>::quiet_NaN();
    return static_cast<float>(num/den);
}

void cube2mat_hurst_rs_ret(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                           const py::dict& cubes_map) {
    if(!cubes_map.contains("last_price")){
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
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
            std::vector<float> r; r.reserve(d2);
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                if(!(p0>0.f) || !(p1>0.f) || std::isnan(p0) || std::isnan(p1)) continue;
                float lr=std::log(p1) - std::log(p0);
                if(std::isfinite(lr)) r.push_back(lr);
            }
            if(r.size()<20){
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            }else{
                res_ptr[i*d1+j]=hurst_rs(r);
            }
        }
    }
}

void bind_cube2mat_hurst_rs_ret(py::module& m){
    m.def("cube2mat_hurst_rs_ret", &cube2mat_hurst_rs_ret,
          py::arg("result"), py::arg("cubes_map"),
          "R/S Hurst exponent estimated from intraday log returns.");
}
