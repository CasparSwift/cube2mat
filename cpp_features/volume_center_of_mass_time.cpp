#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_volume_center_of_mass_time(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                         const py::dict& cubes_map){
    if(!cubes_map.contains("interval_volume"))
        throw std::runtime_error("cubes_map must contain 'interval_volume'");
    auto vol_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["interval_volume"]);
    auto vbuf=vol_arr.request();
    if(vbuf.ndim!=3) throw std::runtime_error("interval_volume must be 3D");
    const ssize_t d0=vbuf.shape[0], d1=vbuf.shape[1], d2=vbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* vol_ptr=static_cast<float*>(vbuf.ptr); float* res_ptr=static_cast<float*>(rbuf.ptr);
    const double TOTAL_MIN=389.0; const double interval_min=5.0;

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            double wsum=0.0, wtime=0.0;
            for(ssize_t t=0;t<d2;++t){
                float v=vol_ptr[base+t];
                if(std::isnan(v) || !(v>0.f)) continue;
                double frac = (t*interval_min)/TOTAL_MIN;
                wsum += v; wtime += v*frac;
            }
            if(wsum>0.0){ double val=wtime/wsum; if(val<0.0) val=0.0; if(val>1.0) val=1.0; res_ptr[i*d1+j]=static_cast<float>(val); }
            else res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_volume_center_of_mass_time(py::module& m){
    m.def("cube2mat_volume_center_of_mass_time", &cube2mat_volume_center_of_mass_time,
          py::arg("result"), py::arg("cubes_map"),
          "Weighted time centroid by volume in RTH, normalized to [0,1].");
}

