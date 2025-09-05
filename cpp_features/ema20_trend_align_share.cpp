#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_ema20_trend_align_share(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                      const py::dict& cubes_map) {
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
    const double alpha = 2.0/21.0; // span=20

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            double ema=0.0, prev_price=0.0, prev_ema=0.0; bool init=false;
            int align=0, total=0;
            for(ssize_t t=0;t<d2;++t){
                float c=price_ptr[base+t];
                if(std::isnan(c)) continue;
                if(!init){
                    ema=c; prev_price=c; prev_ema=ema; init=true; continue;
                }
                ema = alpha*c + (1.0-alpha)*ema;
                double sc = c - prev_price;
                double se = ema - prev_ema;
                int sign_c = (sc>0)?1:((sc<0)?-1:0);
                int sign_e = (se>0)?1:((se<0)?-1:0);
                if(sign_c!=0 && sign_e!=0){
                    if(sign_c==sign_e) ++align;
                    ++total;
                }
                prev_price=c; prev_ema=ema;
            }
            if(total>0) res_ptr[i*d1+j]=static_cast<float>(align)/static_cast<float>(total);
            else res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_ema20_trend_align_share(py::module& m){
    m.def("cube2mat_ema20_trend_align_share", &cube2mat_ema20_trend_align_share,
          py::arg("result"), py::arg("cubes_map"),
          "Share of RTH bars where sign(Δclose) == sign(ΔEMA20(close)); zeros ignored.");
}
