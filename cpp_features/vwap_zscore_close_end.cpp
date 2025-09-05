#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_vwap_zscore_close_end(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                    const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_vwap") || !cubes_map.contains("interval_volume"))
        throw std::runtime_error("cubes_map must contain 'last_price','interval_vwap','interval_volume'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto vwap_arr =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_vwap"]);
    auto vol_arr  =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_volume"]);
    auto pbuf=price_arr.request(); auto wbuf=vwap_arr.request(); auto vbuf=vol_arr.request();
    if(pbuf.ndim!=3 || wbuf.ndim!=3 || vbuf.ndim!=3) throw std::runtime_error("inputs must be 3D");
    if(pbuf.shape[0]!=wbuf.shape[0] || pbuf.shape[1]!=wbuf.shape[1] || pbuf.shape[2]!=wbuf.shape[2] ||
       vbuf.shape[0]!=pbuf.shape[0] || vbuf.shape[1]!=pbuf.shape[1] || vbuf.shape[2]!=pbuf.shape[2])
        throw std::runtime_error("arrays must have same shape");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    const float* vwap_ptr =static_cast<float*>(wbuf.ptr);
    const float* vol_ptr  =static_cast<float*>(vbuf.ptr); float* res_ptr=static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            double vwap_num=0.0, vwap_den=0.0;
            double sum=0.0, sumsq=0.0; int cnt=0;
            for(ssize_t t=0;t<d2;++t){
                float c=price_ptr[base+t];
                float w=vwap_ptr[base+t];
                float v=vol_ptr[base+t];
                if(!std::isnan(w) && !std::isnan(v) && v>0.f){ vwap_num+=w*v; vwap_den+=v; }
                if(!std::isnan(c) && !std::isnan(w)){
                    double d=static_cast<double>(c)-static_cast<double>(w);
                    sum+=d; sumsq+=d*d; ++cnt;
                }
            }
            float last_c=price_ptr[base+d2-1];
            if(cnt>=2 && vwap_den>0.0 && !std::isnan(last_c)){
                double mean=sum/cnt;
                double var=(sumsq - sum*sum/cnt)/(cnt-1);
                double sd = (var>0.0)? std::sqrt(var) : -1.0;
                double vw=vwap_num/vwap_den;
                if(sd>0.0)
                    res_ptr[i*d1+j]=static_cast<float>((static_cast<double>(last_c)-vw)/sd);
                else res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            }else{
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_vwap_zscore_close_end(py::module& m){
    m.def("cube2mat_vwap_zscore_close_end", &cube2mat_vwap_zscore_close_end,
          py::arg("result"), py::arg("cubes_map"),
          "Z-score of last close vs session VWAP using std(close - vwap) as scale.");
}

