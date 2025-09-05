#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_roll_spread(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            std::vector<float> dp; dp.reserve(d2);
            float prev=std::numeric_limits<float>::quiet_NaN();
            for(ssize_t t=0;t<d2;++t){
                float p=price_ptr[base+t];
                if(!(p>0.f) || std::isnan(p)) { prev=std::numeric_limits<float>::quiet_NaN(); continue; }
                if(!std::isnan(prev)) dp.push_back(p - prev);
                prev=p;
            }
            size_t m=dp.size();
            if(m<3){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<float> x(dp.begin()+1, dp.end());
            std::vector<float> y(dp.begin(), dp.end()-1);
            size_t n=x.size();
            double mean_x=0.0, mean_y=0.0;
            for(size_t k=0;k<n;++k){ mean_x+=x[k]; mean_y+=y[k]; }
            mean_x/=n; mean_y/=n;
            double cov=0.0;
            for(size_t k=0;k<n;++k){ cov += (x[k]-mean_x)*(y[k]-mean_y); }
            cov /= (n>1)? (n-1) : 1;
            if(std::isfinite(cov) && cov<0.0) res_ptr[i*d1+j]=static_cast<float>(2.0*std::sqrt(-cov));
            else res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_roll_spread(py::module& m){
    m.def("cube2mat_roll_spread", &cube2mat_roll_spread,
          py::arg("result"), py::arg("cubes_map"),
          "Roll effective spread estimator from lag-1 autocovariance of price changes within RTH.");
}

