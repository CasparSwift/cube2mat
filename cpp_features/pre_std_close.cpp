#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_pre_std_close(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
    const ssize_t RTH_START=114;

    #pragma omp parallel for collapse(2)
    for (ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            if (d2 < RTH_START){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double sum=0.0,sum2=0.0; int cnt=0;
            for (ssize_t t=0; t<RTH_START; ++t){
                float c=price_ptr[base+t];
                if (std::isnan(c)) continue;
                sum+=c; sum2+=c*c; ++cnt;
            }
            if (cnt>=3){
                double var=(sum2 - sum*sum/cnt)/(cnt-1);
                res_ptr[i*d1+j]=(var>0.0)?static_cast<float>(std::sqrt(var)):std::numeric_limits<float>::quiet_NaN();
            }else{
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void bind_cube2mat_pre_std_close(py::module& m){
    m.def("cube2mat_pre_std_close", &cube2mat_pre_std_close,
          py::arg("result"), py::arg("cubes_map"),
          "Pre-market std(close) between 00:00–09:29; NaN if <3 ticks.");
}

