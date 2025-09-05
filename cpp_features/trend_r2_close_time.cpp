#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_trend_r2_close_time(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                  const py::dict& cubes_map){
    if(!cubes_map.contains("last_price"))
        throw std::runtime_error("cubes_map must contain 'last_price'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf=price_arr.request();
    if(pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1)
        throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr); float* res_ptr=static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<float> y; y.reserve(d2);
            for(ssize_t t=0;t<d2;++t){ float c=price_ptr[base+t]; if(!std::isnan(c)) y.push_back(c); }
            size_t n=y.size();
            if(n<2){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double y_sum=0.0; for(float v: y) y_sum+=v; double y_mean=y_sum/n;
            double sst=0.0; for(float v: y){ double d=v-y_mean; sst+=d*d; }
            if(!(sst>0.0)){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double x_mean=(n-1)/2.0; double sxx=0.0; double sxy=0.0;
            for(size_t k=0;k<n;++k){ double xd=k-x_mean; sxx+=xd*xd; sxy+=xd*(y[k]-y_mean); }
            if(!(sxx>0.0)){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double cov=sxy; double r2=(cov*cov)/(sxx*sst);
            res_ptr[i*d1+j]=static_cast<float>(r2);
        }
    }
}

void bind_cube2mat_trend_r2_close_time(py::module& m){
    m.def("cube2mat_trend_r2_close_time", &cube2mat_trend_r2_close_time,
          py::arg("result"), py::arg("cubes_map"),
          "R^2 of OLS close~time (minutes since 09:30) within 09:30–15:59; NaN if <2 points or var(close)=0.");
}

