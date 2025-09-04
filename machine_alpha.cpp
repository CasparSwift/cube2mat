#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_alpha_example(py::module& m);


PYBIND11_MODULE(machine_alpha, m) {
    m.doc() = "collection of intraday features";
    bind_alpha_example(m);
}