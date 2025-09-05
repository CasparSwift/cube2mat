#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_duration_since_last_vwap_cross_frac(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                                  const py::dict& cubes_map) {
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_vwap")){
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_vwap'");
    }
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto vwap_arr =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_vwap"]);
    auto pbuf=price_arr.request(); auto wbuf=vwap_arr.request();
    if(pbuf.ndim!=3 || wbuf.ndim!=3 || pbuf.shape[0]!=wbuf.shape[0] || pbuf.shape[1]!=wbuf.shape[1] || pbuf.shape[2]!=wbuf.shape[2])
        throw std::runtime_error("arrays must have same 3D shape");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    const float* vwap_ptr =static_cast<float*>(wbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            int last_cross=-1;
            float prev_diff=0.0f; bool prev_set=false;
            for(ssize_t t=0;t<d2;++t){
                float c=price_ptr[base+t];
                float w=vwap_ptr[base+t];
                if(std::isnan(c)||std::isnan(w)) continue;
                float diff=c-w;
                if(prev_set){
                    if((diff>0 && prev_diff<0) || (diff<0 && prev_diff>0)){
                        last_cross=t;
                    }
                }
                prev_diff=diff; prev_set=true;
            }
            if(last_cross<0){
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            }else{
                double frac = (d2-1>0)? static_cast<double>(d2-1-last_cross)/static_cast<double>(d2-1) : std::numeric_limits<double>::quiet_NaN();
                res_ptr[i*d1+j]=static_cast<float>(frac);
            }
        }
    }
}

void bind_cube2mat_duration_since_last_vwap_cross_frac(py::module& m){
    m.def("cube2mat_duration_since_last_vwap_cross_frac", &cube2mat_duration_since_last_vwap_cross_frac,
          py::arg("result"), py::arg("cubes_map"),
          "Session fraction since last VWAP crossing (stickiness).");
}
