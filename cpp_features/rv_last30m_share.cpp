#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_rv_last30m_share(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                               const py::dict& cubes_map){
    if(!cubes_map.contains("last_price")){
        throw std::runtime_error("cubes_map must contain 'last_price'");
    }
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if(pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

    const double TOTAL_MIN=389.0;
    const double interval_min = (d2>1)? TOTAL_MIN/static_cast<double>(d2-1) : TOTAL_MIN;
    ssize_t k = static_cast<ssize_t>(std::ceil(30.0/interval_min));
    if(k<1) k=1; ssize_t n_ret=d2-1; ssize_t start_idx = (n_ret>k)? n_ret - k + 1 : 1;

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            double total=0.0; double last=0.0; int cnt=0; int cnt_last=0;
            for(ssize_t t=1;t<d2;++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                if(std::isnan(p0)||std::isnan(p1)||!(p0>0.f)||!(p1>0.f)) continue;
                double r=std::log(static_cast<double>(p1)/p0); if(!std::isfinite(r)) continue;
                double r2=r*r; total+=r2; ++cnt;
                if(t>=start_idx){ last+=r2; ++cnt_last; }
            }
            if(!(total>0.0) || cnt==0 || cnt_last==0){
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            }else{
                res_ptr[i*d1+j]=static_cast<float>(last/total);
            }
        }
    }
}

void bind_cube2mat_rv_last30m_share(py::module& m){
    m.def("cube2mat_rv_last30m_share", &cube2mat_rv_last30m_share,
          py::arg("result"), py::arg("cubes_map"),
          "Share of realized variance contributed by the last 30 minutes of the session.");
}

