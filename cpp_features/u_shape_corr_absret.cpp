#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_u_shape_corr_absret(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                  const py::dict& cubes_map){
    if(!cubes_map.contains("last_price"))
        throw std::runtime_error("cubes_map must contain 'last_price'");
    auto price_arr=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf=price_arr.request();
    if(pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1)
        throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr); float* res_ptr=static_cast<float*>(rbuf.ptr);

#pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<double> r; r.reserve(d2);
            float prev=std::numeric_limits<float>::quiet_NaN();
            for(ssize_t t=0;t<d2;++t){
                float c=price_ptr[base+t];
                if(std::isnan(c) || !(c>0.f)) { prev=std::numeric_limits<float>::quiet_NaN(); continue; }
                if(!std::isnan(prev) && prev>0.f){ r.push_back(std::fabs(std::log(c/prev))); }
                prev=c;
            }
            size_t n=r.size();
            if(n<3){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double mean_x=0.0; for(double v: r) mean_x+=v; mean_x/=n;
            std::vector<double> u(n); double sum_u=0.0; double denom = (n>1)?(n-1):1;
            for(size_t k=0;k<n;++k){ double tfrac = (denom>0)?(static_cast<double>(k)/denom):0.0; double val=(tfrac-0.5); double sq=val*val; u[k]=sq; sum_u+=sq; }
            double mean_u=sum_u/n; double var_u=0.0; for(size_t k=0;k<n;++k){ double d=u[k]-mean_u; var_u+=d*d; }
            double std_u = (n>1)?std::sqrt(var_u/(n-1)):0.0; double su=0.0;
            for(size_t k=0;k<n;++k){ u[k]=(u[k]-mean_u)/(std_u>0?std_u:1.0); su+=u[k]*u[k]; }
            su=std::sqrt(su); if(!(su>0.0)){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double sx=0.0; double dot=0.0;
            for(size_t k=0;k<n;++k){ double xd=r[k]-mean_x; sx+=xd*xd; dot+=xd*u[k]; }
            if(!(sx>0.0)){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            res_ptr[i*d1+j]=static_cast<float>(dot/(std::sqrt(sx)*su));
        }
    }
}

void bind_cube2mat_u_shape_corr_absret(py::module& m){
    m.def("cube2mat_u_shape_corr_absret", &cube2mat_u_shape_corr_absret,
          py::arg("result"), py::arg("cubes_map"),
          "Correlation of |log return| with U-shape time template.");
}

