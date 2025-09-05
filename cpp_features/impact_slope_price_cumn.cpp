#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_impact_slope_price_cumn(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            float anchor = price_ptr[base];
            if(std::isnan(anchor)) { res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double sx=0.0, sy=0.0, sxx=0.0, sxy=0.0; int cnt=0;
            for(ssize_t t=0;t<d2;++t){
                float p=price_ptr[base+t];
                if(std::isnan(p)) continue;
                double x = static_cast<double>(t+1); // cumulative bar count
                double y = static_cast<double>(p - anchor);
                sx+=x; sy+=y; sxx+=x*x; sxy+=x*y; ++cnt;
            }
            if(cnt>1){
                double den = sxx - sx*sx/cnt;
                double num = sxy - sx*sy/cnt;
                res_ptr[i*d1+j]=(den>0.0)?static_cast<float>(num/den):std::numeric_limits<float>::quiet_NaN();
            }else{
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_impact_slope_price_cumn(py::module& m){
    m.def("cube2mat_impact_slope_price_cumn", &cube2mat_impact_slope_price_cumn,
          py::arg("result"), py::arg("cubes_map"),
          "OLS beta of (close - first_open) on cumulative n (RTH)." );
}
