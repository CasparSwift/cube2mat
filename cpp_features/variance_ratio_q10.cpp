#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_variance_ratio_q10(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                 const py::dict& cubes_map){
    if(!cubes_map.contains("last_price"))
        throw std::runtime_error("cubes_map must contain 'last_price'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf=price_arr.request();
    if(pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr); float* res_ptr=static_cast<float*>(rbuf.ptr);
    const int q=10;

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<double> r; r.reserve(d2);
            float prev=std::numeric_limits<float>::quiet_NaN();
            for(ssize_t t=0;t<d2;++t){ float c=price_ptr[base+t]; if(std::isnan(c) || !(c>0.f)){ prev=std::numeric_limits<float>::quiet_NaN(); continue; } if(!std::isnan(prev) && prev>0.f) r.push_back(std::log(c/prev)); prev=c; }
            size_t n=r.size(); if(n<static_cast<size_t>(q)+2){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double mean=0.0; for(double v: r) mean+=v; mean/=n; double var1=0.0; for(double v: r){ double d=v-mean; var1+=d*d; } var1/= (n>1? (n-1):1);
            if(!(var1>0.0)){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<double> roll; roll.reserve(n-q+1); double s=0.0; for(int k=0;k<q;++k) s+=r[k]; roll.push_back(s); for(size_t k=q;k<n;++k){ s+=r[k]-r[k-q]; roll.push_back(s); }
            size_t m=roll.size(); if(m<2){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double meanr=0.0; for(double v: roll) meanr+=v; meanr/=m; double varq=0.0; for(double v: roll){ double d=v-meanr; varq+=d*d; } varq/= (m>1? (m-1):1);
            res_ptr[i*d1+j]=static_cast<float>((varq>=0.0)?(varq/(q*var1)):std::numeric_limits<float>::quiet_NaN());
        }
    }
}

void bind_cube2mat_variance_ratio_q10(py::module& m){
    m.def("cube2mat_variance_ratio_q10", &cube2mat_variance_ratio_q10,
          py::arg("result"), py::arg("cubes_map"),
          "Variance Ratio of intraday log returns with q=10 in 09:30–15:59.");
}

