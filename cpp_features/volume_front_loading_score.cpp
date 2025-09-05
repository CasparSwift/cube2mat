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

void cube2mat_volume_front_loading_score(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                         const py::dict& cubes_map){
    if(!cubes_map.contains("interval_volume"))
        throw std::runtime_error("cubes_map must contain 'interval_volume'");
    auto vol_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf=vol_arr.request();
    if(vbuf.ndim!=3) throw std::runtime_error("interval_volume must be 3D");
    const ssize_t d0=vbuf.shape[0], d1=vbuf.shape[1], d2=vbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result mismatch");
    const float* vol_ptr=static_cast<float*>(vbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            ssize_t base=i*(d1*d2)+j*d2;
            double tot=0.0; std::vector<float> vols(d2,0.0f);
            for(ssize_t t=0;t<d2;++t){
                float v=vol_ptr[base+t];
                if(std::isfinite(v) && v>0.0f){ vols[t]=v; tot+=v; }
            }
            if(tot<=0.0){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double cum=0.0; double auc=0.0; double prev_y=0.0;
            for(ssize_t t=0;t<d2;++t){
                cum+=vols[t];
                double y=cum/tot;
                double x_step=1.0/d2;
                auc += (prev_y + y)*0.5*x_step;
                prev_y=y;
            }
            res_ptr[i*d1+j]=static_cast<float>(2*auc - 1.0);
        }
    }
}

void bind_cube2mat_volume_front_loading_score(py::module& m){
    m.def("cube2mat_volume_front_loading_score", &cube2mat_volume_front_loading_score,
          py::arg("result"), py::arg("cubes_map"),
          "2*AUC(cumVolFraction vs timeFraction) - 1; positive=front-loaded");
}

