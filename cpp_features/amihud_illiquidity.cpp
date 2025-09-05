#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_amihud_illiquidity(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                 const py::dict& cubes_map) {
    if (!cubes_map.contains("last_price") || !cubes_map.contains("interval_vwap") || !cubes_map.contains("interval_volume")) {
        throw std::runtime_error("cubes_map must contain 'last_price','interval_vwap','interval_volume'");
    }
    auto price_arr  = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto wap_arr   = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_vwap"]);
    auto vol_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["interval_volume"]);

    auto cbuf = price_arr.request();
    auto wbuf = wap_arr.request();
    auto vbuf = vol_arr.request();
    if (cbuf.ndim !=3 || wbuf.ndim !=3 || vbuf.ndim !=3) {
        throw std::runtime_error("arrays must be 3D");
    }
    if (cbuf.shape[0]!=wbuf.shape[0] || cbuf.shape[1]!=wbuf.shape[1] || cbuf.shape[2]!=wbuf.shape[2] ||
        cbuf.shape[0]!=vbuf.shape[0] || cbuf.shape[1]!=vbuf.shape[1] || cbuf.shape[2]!=vbuf.shape[2]) {
        throw std::runtime_error("arrays must have same shape");
    }
    const ssize_t d0=cbuf.shape[0], d1=cbuf.shape[1], d2=cbuf.shape[2];
    auto rbuf = result.request();
    if (rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) {
        throw std::runtime_error("result must be (d0,d1)");
    }
    const float* price_ptr = static_cast<float*>(cbuf.ptr);
    const float* wap_ptr  = static_cast<float*>(wbuf.ptr);
    const float* vol_ptr   = static_cast<float*>(vbuf.ptr);
    float* res_ptr         = static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for (ssize_t i=0;i<d0;++i){
        for (ssize_t j=0;j<d1;++j){
            ssize_t base=i*(d1*d2)+j*d2;
            float sum=0.0f; ssize_t cnt=0;
            for (ssize_t t=1;t<d2;++t){
                float c0=price_ptr[base+t-1];
                float c1=price_ptr[base+t];
                float w =wap_ptr [base+t];
                float v =vol_ptr  [base+t];
                if(!std::isfinite(c0)||!std::isfinite(c1)||!std::isfinite(w)||!std::isfinite(v)||w<=0.0f||v<=0.0f||c0<=0.0f) continue;
                float ret=(c1/c0)-1.0f;
                sum += std::fabs(ret)/(w*v);
                ++cnt;
            }
            res_ptr[i*d1+j]=(cnt>0)?(sum/cnt):std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_amihud_illiquidity(py::module& m){
    m.def("cube2mat_amihud_illiquidity", &cube2mat_amihud_illiquidity,
          py::arg("result"), py::arg("cubes_map"),
          "Amihud illiquidity mean(|ret|/(interval_vwap*interval_volume)).");
}

