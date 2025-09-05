#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_run_count_total(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                              const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price")) throw std::runtime_error("cubes_map must contain 'last_price'");
    auto price_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf = price_arr.request();
    if (pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr = static_cast<float*>(pbuf.ptr);
    float* res_ptr = static_cast<float*>(rbuf.ptr);
    ssize_t rth_start=0,rth_end=d2-1; if(d2>=288){rth_start=114;rth_end=191;}

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            int runs=0; int prev=0; int nonzero=0;
            for(ssize_t t=rth_start+1; t<=rth_end; ++t){
                float p0=price_ptr[base+t-1];
                float p1=price_ptr[base+t];
                if(std::isnan(p0) || std::isnan(p1) || !(p0>0.f) || !(p1>0.f)) { prev=0; continue; }
                double r=std::log(p1/p0);
                int sign = (r>0)?1:((r<0)?-1:0);
                if(sign==0){ prev=0; continue; }
                ++nonzero;
                if(prev==0 || sign!=prev) ++runs;
                prev=sign;
            }
            res_ptr[i*d1+j]=(nonzero>0)? static_cast<float>(runs)
                                     : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_run_count_total(py::module& m){
    m.def("cube2mat_run_count_total", &cube2mat_run_count_total,
          py::arg("result"), py::arg("cubes_map"),
          "Total number of sign runs in intraday log-returns (zeros break runs).");
}

