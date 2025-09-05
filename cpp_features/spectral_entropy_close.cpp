#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_spectral_entropy_close(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
                                     const py::dict& cubes_map){
    if(!cubes_map.contains("last_price")) throw std::runtime_error("cubes_map must contain 'last_price'");
    auto price_arr=py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(cubes_map["last_price"]);
    auto pbuf=price_arr.request();
    if(pbuf.ndim!=3) throw std::runtime_error("last_price must be 3D");
    const ssize_t d0=pbuf.shape[0], d1=pbuf.shape[1], d2=pbuf.shape[2];
    auto rbuf=result.request();
    if(rbuf.ndim!=2 || rbuf.shape[0]!=d0 || rbuf.shape[1]!=d1) throw std::runtime_error("result must be (d0,d1)");
    const float* price_ptr=static_cast<float*>(pbuf.ptr);
    float* res_ptr=static_cast<float*>(rbuf.ptr);

    const double PI = std::acos(-1.0);

    #pragma omp parallel for collapse(2)
    for(ssize_t i=0;i<d0;++i){
        for(ssize_t j=0;j<d1;++j){
            const ssize_t base=i*(d1*d2)+j*d2;
            std::vector<double> x; x.reserve(d2);
            bool ok=true;
            for(ssize_t t=0;t<d2;++t){
                float c=price_ptr[base+t];
                if(std::isnan(c)) { ok=false; break; }
                x.push_back(c);
            }
            size_t n=x.size();
            if(!ok || n<2){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            // detrend linear
            double sx=0.0, sy=0.0, sxx=0.0, sxy=0.0;
            for(size_t t=0;t<n;++t){ sx+=t; sy+=x[t]; sxx+=t*t; sxy+=t*x[t]; }
            double denom=n*sxx - sx*sx;
            double slope=0.0, intercept=0.0;
            if(std::fabs(denom)>1e-12){ slope=(n*sxy - sx*sy)/denom; intercept=(sy - slope*sx)/n; }
            std::vector<double> r; r.reserve(n);
            for(size_t t=0;t<n;++t){ r.push_back(x[t] - (intercept + slope*t)); }
            size_t nfreq = n/2 + 1;
            if(nfreq<=1){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::vector<double> ps(nfreq,0.0);
            double total=0.0;
            for(size_t k=0;k<nfreq;++k){
                std::complex<double> acc(0.0,0.0);
                for(size_t t=0;t<n;++t){
                    double theta=-2.0*PI*static_cast<double>(k)*static_cast<double>(t)/static_cast<double>(n);
                    acc += std::complex<double>(r[t]*std::cos(theta), r[t]*std::sin(theta));
                }
                double power=std::norm(acc);
                ps[k]=power; total+=power;
            }
            if(!(total>0.0)){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            double ent=0.0; for(size_t k=0;k<nfreq;++k){ double p=ps[k]/total; if(p>0.0) ent -= p*std::log(p); }
            ent /= std::log(static_cast<double>(nfreq));
            res_ptr[i*d1+j]=static_cast<float>(ent);
        }
    }
}

void bind_cube2mat_spectral_entropy_close(py::module& m){
    m.def("cube2mat_spectral_entropy_close", &cube2mat_spectral_entropy_close,
          py::arg("result"), py::arg("cubes_map"),
          "Normalized spectral entropy (0..1) of detrended close.");
}

