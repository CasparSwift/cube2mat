#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void cube2mat_price_above_median_share(py::array_t<float, py::array::c_style | py::array::forcecast>& result,
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
            std::vector<float> vals; vals.reserve(rth_end-rth_start+1);
            for(ssize_t t=rth_start; t<=rth_end; ++t){
                float c=price_ptr[base+t];
                if(!std::isnan(c)) vals.push_back(c);
            }
            if(vals.empty()){ res_ptr[i*d1+j]=std::numeric_limits<float>::quiet_NaN(); continue; }
            std::sort(vals.begin(), vals.end());
            double median;
            size_t n=vals.size();
            if(n%2==1) median=vals[n/2];
            else median=0.5*(vals[n/2-1]+vals[n/2]);
            int above=0, valid=0;
            for(ssize_t t=rth_start; t<=rth_end; ++t){
                float c=price_ptr[base+t];
                if(std::isnan(c)) continue; ++valid; if(c>median) ++above;
            }
            res_ptr[i*d1+j]=(valid>0)? static_cast<float>(static_cast<double>(above)/valid)
                                     : std::numeric_limits<float>::quiet_NaN();
        }
    }
}

void bind_cube2mat_price_above_median_share(py::module& m){
    m.def("cube2mat_price_above_median_share", &cube2mat_price_above_median_share,
          py::arg("result"), py::arg("cubes_map"),
          "Share of RTH bars with close strictly above the RTH median close.");
}

