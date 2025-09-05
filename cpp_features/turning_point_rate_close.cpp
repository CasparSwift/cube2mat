#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_turning_point_rate_close(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            std::vector<float> s; s.reserve(d2);
            for(ssize_t t=0;t<d2;++t){ float c=price_ptr[base+t]; if(!std::isnan(c)) s.push_back(c); }
            size_t n=s.size();
            if(n<3){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<double> d; d.reserve(n-1);
            for(size_t k=1;k<n;++k) d.push_back(static_cast<double>(s[k]-s[k-1]));
            size_t interior = n-2; if(interior==0){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            size_t cnt=0; for(size_t k=0;k<interior;++k){ double s1=d[k]; double s2=d[k+1]; if(s1==0 || s2==0) continue; if(s1*s2<0) cnt++; }
            res_ptr[i*d1+j]=static_cast<float>(static_cast<double>(cnt)/interior);
        }
    }
}

void bind_cube2mat_turning_point_rate_close(py::module& m){
    m.def("cube2mat_turning_point_rate_close", &cube2mat_turning_point_rate_close,
          py::arg("result"), py::arg("cubes_map"),
          "Fraction of interior RTH bars that are local turning points of close (sign of Δclose flips; endpoints excluded).");
}

