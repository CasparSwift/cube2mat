#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_intraday_max_drawup_close(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            float run_min=std::numeric_limits<float>::infinity();
            float maxdu=0.0f; int count=0;
            for(ssize_t t=0;t<d2;++t){
                float p=price_ptr[base+t];
                if(std::isnan(p) || !(p>0.f)) continue;
                if(!std::isfinite(run_min)) run_min=p;
                if(p<run_min) run_min=p;
                if(run_min>0.f){
                    float du = p/run_min - 1.0f;
                    if(std::isfinite(du) && du>maxdu) maxdu=du;
                }
                ++count;
            }
            if(count<2) res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            else res_ptr[i*d1+j]=maxdu;
        }
    }
}

void bind_cube2mat_intraday_max_drawup_close(py::module& m){
    m.def("cube2mat_intraday_max_drawup_close", &cube2mat_intraday_max_drawup_close,
          py::arg("result"), py::arg("cubes_map"),
          "Max drawup using close (RTH).");
}
