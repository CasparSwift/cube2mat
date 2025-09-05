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

void cube2mat_vwap_gap_tail_share_q75(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                      const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_vwap"))
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_vwap'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto vwap_arr =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_vwap"]);
    auto pbuf=price_arr.request(); auto wbuf=vwap_arr.request();
    if(pbuf.ndim!=3 || wbuf.ndim!=3) throw std::runtime_error("inputs must be 3D");
    if(pbuf.shape[0]!=wbuf.shape[0] || pbuf.shape[1]!=wbuf.shape[1] || pbuf.shape[2]!=wbuf.shape[2])
        throw std::runtime_error("last_price and interval_vwap must have same shape");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    const float* vwap_ptr =static_cast<float*>(wbuf.ptr); float* res_ptr=static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<double> gaps; gaps.reserve(d2);
            for(ssize_t t=0;t<d2;++t){
                float c=price_ptr[base+t]; float w=vwap_ptr[base+t];
                if(std::isnan(c) || std::isnan(w)) continue;
                double g=std::fabs(static_cast<double>(c)-static_cast<double>(w));
                if(std::isfinite(g)) gaps.push_back(g);
            }
            size_t n=gaps.size();
            if(n<3){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::sort(gaps.begin(), gaps.end());
            double thresh=gaps[static_cast<size_t>(0.75*(n-1))];
            size_t cnt=0; for(double g: gaps) if(g>thresh) ++cnt;
            res_ptr[i*d1+j]=static_cast<float>(static_cast<double>(cnt)/n);
        }
    }
}

void bind_cube2mat_vwap_gap_tail_share_q75(py::module& m){
    m.def("cube2mat_vwap_gap_tail_share_q75", &cube2mat_vwap_gap_tail_share_q75,
          py::arg("result"), py::arg("cubes_map"),
          "Share of bars where |close - vwap| exceeds its 75th percentile; NaN if <3 bars.");
}

