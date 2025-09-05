#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static float slope_norm(const std::vector<float>& y){
    size_t n=y.size();
    if(n<2) return std::numeric_limits<float>::quiet_NaN();
    double y_sum=0.0; for(float v: y) y_sum+=v; double y_mean=y_sum/n;
    double x_mean=(n-1)/2.0; double sxx=0.0; double sxy=0.0;
    for(size_t k=0;k<n;++k){
        double xd=k-x_mean; double yd=y[k]-y_mean; sxx+=xd*xd; sxy+=xd*yd;
    }
    if(!(sxx>0.0)) return std::numeric_limits<float>::quiet_NaN();
    double slope_x=sxy/sxx; // per index
    return static_cast<float>(slope_x*(n-1));
}

void cube2mat_trend_piecewise_slope_am_pm(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                          const py::dict& cubes_map,
                                          ssize_t idx_1200){
    if(!cubes_map.contains("last_price"))
        throw std::runtime_error("cubes_map must contain 'last_price'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf=price_arr.request();
    if(pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    if(idx_1200<0 || idx_1200>=d2) throw std::runtime_error("idx_1200 out of range");
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1)
        throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<float> am; am.reserve(idx_1200+1);
            std::vector<float> pm; pm.reserve(d2-idx_1200);
            for(ssize_t t=0;t<=idx_1200;++t){
                float c=price_ptr[base+t];
                if(!std::isnan(c)) am.push_back(c);
            }
            for(ssize_t t=idx_1200;t<d2;++t){
                float c=price_ptr[base+t];
                if(!std::isnan(c)) pm.push_back(c);
            }
            float sa=slope_norm(am);
            float sp=slope_norm(pm);
            if(std::isnan(sa)||std::isnan(sp))
                res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN();
            else
                res_ptr[i*d1+j]=sp-sa;
        }
    }
}

void bind_cube2mat_trend_piecewise_slope_am_pm(py::module& m){
    m.def("cube2mat_trend_piecewise_slope_am_pm", &cube2mat_trend_piecewise_slope_am_pm,
          py::arg("result"), py::arg("cubes_map"), py::arg("idx_1200"),
          "Afternoon minus morning OLS slope of close~time within RTH.");
}

