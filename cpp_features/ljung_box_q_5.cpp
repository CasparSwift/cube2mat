#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static double acf_at_lag(const std::vector<double>& r, int k){
    int n=r.size();
    if(n<k+2) return std::numeric_limits<double>::quiet_NaN();
    double mean=0.0; for(double v: r) mean+=v; mean/=n;
    double den=0.0; for(double v: r){ double d=v-mean; den+=d*d; }
    if(!(den>0.0) || !std::isfinite(den)) return std::numeric_limits<double>::quiet_NaN();
    double num=0.0; for(int t=k;t<n;++t){ num += (r[t]-mean)*(r[t-k]-mean); }
    return num/den;
}

void cube2mat_ljung_box_q_5(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                            const py::dict& cubes_map){
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
    const int m=5;

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<double> r; r.reserve(d2);
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                if(!(p0>0.f) || !(p1>0.f) || std::isnan(p0) || std::isnan(p1)) continue;
                double lr=std::log(p1) - std::log(p0);
                if(std::isfinite(lr)) r.push_back(lr);
            }
            int n=r.size();
            if(n<=m){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double q=0.0; bool valid=true;
            for(int k=1;k<=m;++k){
                double rho=acf_at_lag(r,k);
                if(!std::isfinite(rho) || (n-k)<=0){ valid=false; break; }
                q += (rho*rho)/ (n - k);
            }
            if(!valid){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); }
            else res_ptr[i*d1+j]=static_cast<float>(n*(n+2)*q);
        }
    }
}

void bind_cube2mat_ljung_box_q_5(py::module& m){
    m.def("cube2mat_ljung_box_q_5", &cube2mat_ljung_box_q_5,
          py::arg("result"), py::arg("cubes_map"),
          "Ljung-Box Q statistic at lags 1..5 for intraday log returns.");
}
