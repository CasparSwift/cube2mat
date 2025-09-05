#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_volume_top3_share(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                const py::dict& cubes_map){
    if(!cubes_map.contains("interval_volume")) throw std::runtime_error("cubes_map must contain 'interval_volume'");
    auto v_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf=v_arr.request();
    if(vbuf.ndim!=3) throw std::runtime_error("interval_volume must be 3D");
    const ssize_t d0=vbuf.shape[0], d1=vbuf.shape[1], d2=vbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2||rbuf.shape[0]!=d0||rbuf.shape[1]!=d1) throw std::runtime_error("result mismatch");
    const float* v_ptr=static_cast<float*>(vbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);
#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            ssize_t base=i*(d1*d2)+j*d2;
            std::vector<float> vols; vols.reserve(d2);
            for(ssize_t t=0;t<d2;++t){
                float v=v_ptr[base+t];
                if(std::isfinite(v)) vols.push_back(v);
            }
            if(vols.empty()){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            float tot=0.0f; for(float v:vols) tot+=v;
            if(!(tot>0)) { res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::sort(vols.begin(), vols.end(), std::greater<float>());
            float top3=0.0f; for(size_t k=0;k<std::min<size_t>(3,vols.size());++k) top3+=vols[k];
            res_ptr[i*d1+j]=top3/tot;
        }
    }
}

void bind_cube2mat_volume_top3_share(py::module& m){
    m.def("cube2mat_volume_top3_share", &cube2mat_volume_top3_share,
          py::arg("result"), py::arg("cubes_map"),
          "Share of total interval_volume from top-3 interval_volume bars");
}

