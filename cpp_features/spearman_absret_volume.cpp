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

static void rankdata(const std::vector<double>& data, std::vector<double>& ranks){
    size_t n = data.size();
    ranks.resize(n);
    std::vector<std::pair<double,size_t>> idx(n);
    for(size_t i=0;i<n;++i) idx[i]={data[i],i};
    std::sort(idx.begin(), idx.end(),[](auto&a,auto&b){return a.first<b.first;});
    size_t i=0;
    while(i<n){
        size_t j=i;
        while(j<n && idx[j].first==idx[i].first) ++j;
        double r = 0.5*(i+j-1)+1.0; // average rank
        for(size_t k=i;k<j;++k) ranks[idx[k].second]=r;
        i=j;
    }
}

void cube2mat_spearman_absret_volume(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                     const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_volume")){
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_volume'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto vol_arr   = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);
    auto pbuf = price_arr.request();
    auto vbuf = vol_arr.request();
    if(pbuf.ndim!=3 || vbuf.ndim!=3) throw std::runtime_error("inputs must be 3D arrays");
    if(pbuf.shape[0]!=vbuf.shape[0]||pbuf.shape[1]!=vbuf.shape[1]||pbuf.shape[2]!=vbuf.shape[2])
        throw std::runtime_error("arrays must have same shape");
    const ssize_t d0=pbuf.shape[0];
    const ssize_t d1=pbuf.shape[1];
    const ssize_t d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2||rbuf.shape[0]!=d0||rbuf.shape[1]!=d1) throw std::runtime_error("result must be 2D");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    const float* vol_ptr=static_cast<float*>(vbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<double> xvec; std::vector<double> yvec;
            xvec.reserve(d2); yvec.reserve(d2);
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                float v = vol_ptr[base+t];
                if(std::isnan(p0)||std::isnan(p1)||std::isnan(v)||!(v>0.f)) continue;
                float r = std::log(p1)-std::log(p0);
                if(std::isnan(r)) continue;
                xvec.push_back(std::fabs(r));
                yvec.push_back(v);
            }
            size_t n=xvec.size();
            if(n<2){
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            std::vector<double> rx, ry;
            rankdata(xvec, rx);
            rankdata(yvec, ry);
            double sx=0.0, sy=0.0, sxx=0.0, syy=0.0, sxy=0.0;
            for(size_t k=0;k<n;++k){
                double x=rx[k], y=ry[k];
                sx+=x; sy+=y; sxx+=x*x; syy+=y*y; sxy+=x*y;
            }
            double cov=sxy - sx*sy/n;
            double varx=sxx - sx*sx/n;
            double vary=syy - sy*sy/n;
            res_ptr[i*d1+j]=(varx>0.0 && vary>0.0)?static_cast<float>(cov/std::sqrt(varx*vary))
                                                   : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_spearman_absret_volume(py::module& m){
    m.def("cube2mat_spearman_absret_volume", &cube2mat_spearman_absret_volume,
          py::arg("result"), py::arg("cubes_map"),
          "Spearman rank correlation between |log returns| and volume in RTH.");
}

