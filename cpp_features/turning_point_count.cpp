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

void cube2mat_turning_point_count(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
    const double theta=0.1;

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<double> closes; closes.reserve(d2);
            for(ssize_t t=0;t<d2;++t){
                float p=price_ptr[base+t];
                if(!std::isnan(p)) closes.push_back(p);
            }
            if(closes.size()<3){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<double> d; d.reserve(closes.size()-1);
            for(size_t t=1;t<closes.size();++t) d.push_back(closes[t]-closes[t-1]);
            std::vector<double> absd(d.size());
            for(size_t t=0;t<d.size();++t) absd[t]=std::fabs(d[t]);
            std::sort(absd.begin(), absd.end());
            double med = absd.size()>0? absd[absd.size()/2] : 0.0;
            if(absd.size()%2==0 && absd.size()>1) med = 0.5*(med + absd[absd.size()/2 -1]);
            double thr = theta * med;
            if(!std::isfinite(thr)) thr = 0.0;
            std::vector<double> d2; d2.reserve(d.size());
            for(double val: d){ if(std::fabs(val) > thr) d2.push_back(val); }
            if(d2.size()<2){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<int> sign; sign.reserve(d2.size());
            for(double val: d2){ if(val>0) sign.push_back(1); else if(val<0) sign.push_back(-1); }
            if(sign.size()<2){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            int flips=0; for(size_t t=1;t<sign.size();++t) if(sign[t]!=sign[t-1]) flips++;
            res_ptr[i*d1+j]=static_cast<float>(flips);
        }
    }
}

void bind_cube2mat_turning_point_count(py::module& m){
    m.def("cube2mat_turning_point_count", &cube2mat_turning_point_count,
          py::arg("result"), py::arg("cubes_map"),
          "Number of local extrema based on filtered sign flips of Δclose (theta=0.1).");
}

