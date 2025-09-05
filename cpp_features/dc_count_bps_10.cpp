#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_dc_count_bps_10(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                              const py::dict& cubes_map){
    if(!cubes_map.contains("last_price"))
        throw std::runtime_error("cubes_map must contain 'last_price'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf=price_arr.request();
    if(pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1)
        throw std::runtime_error("result must be 2D (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);
    const double theta=1e-3; //10 bps

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<float> p; p.reserve(d2);
            for(ssize_t t=0;t<d2;++t){
                float v=price_ptr[base+t];
                if(v>0.f && !std::isnan(v)) p.push_back(v);
            }
            size_t n=p.size();
            if(n<2){
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double count=0.0;
            int dir=0; //0 unknown, +1 up, -1 down
            double extreme=p[0];
            double ref=p[0];
            for(size_t idx=1; idx<n; ++idx){
                double px=p[idx];
                if(dir==0){
                    if(px>=ref*(1.0+theta)){dir=+1; extreme=px; count+=1.0;}
                    else if(px<=ref*(1.0-theta)){dir=-1; extreme=px; count+=1.0;}
                }else if(dir==+1){
                    if(px>extreme) extreme=px;
                    else if(px<=extreme*(1.0-theta)){dir=-1; extreme=px; count+=1.0;}
                }else{ //dir==-1
                    if(px<extreme) extreme=px;
                    else if(px>=extreme*(1.0+theta)){dir=+1; extreme=px; count+=1.0;}
                }
            }
            res_ptr[i*d1+j]=static_cast<float>(count);
        }
    }
}

void bind_cube2mat_dc_count_bps_10(py::module& m){
    m.def("cube2mat_dc_count_bps_10", &cube2mat_dc_count_bps_10,
          py::arg("result"), py::arg("cubes_map"),
          "Directional-change event count at 10bp threshold on close prices.");
}

