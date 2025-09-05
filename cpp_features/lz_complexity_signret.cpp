#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <cmath>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static float lz_complexity_binary(const std::string& seq){
    size_t n=seq.size();
    if(n==0) return std::numeric_limits<float>::quiet_NaN();
    size_t i=0; size_t c=0;
    while(i<n){
        size_t l=1;
        while(i+l<=n){
            std::string sub=seq.substr(i,l);
            if(seq.substr(0,i).find(sub)!=std::string::npos) l++; else break;
        }
        c++; i+=l;
    }
    double norm = (n>1)? (static_cast<double>(n)/std::log2(static_cast<double>(n))) : 1.0;
    return static_cast<float>(c / norm);
}

void cube2mat_lz_complexity_signret(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            std::string seq; seq.reserve(d2);
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                if(!(p0>0.f) || !(p1>0.f) || std::isnan(p0) || std::isnan(p1)) continue;
                float r=std::log(p1/p0);
                if(r>0) seq.push_back('1');
                else if(r<0) seq.push_back('0');
            }
            res_ptr[i*d1+j]=lz_complexity_binary(seq);
        }
    }
}

void bind_cube2mat_lz_complexity_signret(py::module& m){
    m.def("cube2mat_lz_complexity_signret", &cube2mat_lz_complexity_signret,
          py::arg("result"), py::arg("cubes_map"),
          "Lempel-Ziv complexity of the sign of log returns (zeros dropped), normalized by n/log2(n).");
}

