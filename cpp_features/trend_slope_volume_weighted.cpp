#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_trend_slope_volume_weighted(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                          const py::dict& cubes_map){
    if(!cubes_map.contains("last_price") || !cubes_map.contains("interval_volume"))
        throw std::runtime_error("cubes_map must contain 'last_price' and 'interval_volume'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto vol_arr  =py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_volume"]);
    auto pbuf=price_arr.request(); auto vbuf=vol_arr.request();
    if(pbuf.ndim!=3 || vbuf.ndim!=3 || pbuf.shape[0]!=vbuf.shape[0] || pbuf.shape[1]!=vbuf.shape[1] || pbuf.shape[2]!=vbuf.shape[2])
        throw std::runtime_error("arrays must have same 3D shape");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr); const float* vol_ptr=static_cast<float*>(vbuf.ptr); float* res_ptr=static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            double W=0.0; double xw=0.0; double yw=0.0; std::vector<double> xs; xs.reserve(d2); std::vector<double> ys; ys.reserve(d2); std::vector<double> ws; ws.reserve(d2);
            for(ssize_t t=0;t<d2;++t){
                float c=price_ptr[base+t]; float w=vol_ptr[base+t];
                if(std::isnan(c) || std::isnan(w) || !(w>0.0f)) continue;
                double x = static_cast<double>(t);
                xs.push_back(x); ys.push_back(c); ws.push_back(w);
                W += w; xw += w*x; yw += w*c;
            }
            if(W<=0.0 || xs.size()<2){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double mx = xw/W; double my = yw/W; double sxx=0.0, sxy=0.0;
            for(size_t k=0;k<xs.size();++k){ double xd=xs[k]-mx; double yd=ys[k]-my; double w=ws[k]; sxx+=w*xd*xd; sxy+=w*xd*yd; }
            if(!(sxx>0.0)){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            res_ptr[i*d1+j]=static_cast<float>(sxy/sxx); // per index (bars)
        }
    }
}

void bind_cube2mat_trend_slope_volume_weighted(py::module& m){
    m.def("cube2mat_trend_slope_volume_weighted", &cube2mat_trend_slope_volume_weighted,
          py::arg("result"), py::arg("cubes_map"),
          "Volume-weighted OLS slope of close~time (minutes since 09:30) within 09:30–15:59.");
}

