#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
using Dims = std::vector<int64_t>;

class Plan {
 public:
  Plan(std::vector<Dims> shapes, Dims numels)
      : shapes_(std::move(shapes)), numels_(std::move(numels)) {
    if (shapes_.size() != numels_.size()) throw py::value_error("different shape and numel counts");
    for (size_t i = 0; i < shapes_.size(); ++i) {
      int64_t n = 1;
      for (int64_t d : shapes_[i]) {
        if (d <= 0 || d > std::numeric_limits<int64_t>::max() / n) throw py::value_error("invalid shard shape");
        n *= d;
      }
      if (n != numels_[i]) throw py::value_error("shape and numel disagree");
    }
  }

  bool matches(py::handle flat_obj, py::handle shards) const {
    PyObject* seq = shards.ptr();
    // Positive decisions only for exact Python Tensor and a built-in sequence.
    // The caller executes the original Python guard for any unproven case.
    if (Py_TYPE(flat_obj.ptr()) != reinterpret_cast<PyTypeObject*>(THPVariableClass) ||
        (!PyList_CheckExact(seq) && !PyTuple_CheckExact(seq)) ||
        PySequence_Fast_GET_SIZE(seq) != static_cast<Py_ssize_t>(shapes_.size())) return false;
    const auto& flat = THPVariable_Unpack(flat_obj.ptr());
    if (flat.layout() != c10::kStrided || flat.is_nested() ||
        flat.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) return false;
    const auto* root = flat.is_view() ? flat._base().unsafeGetTensorImpl() : flat.unsafeGetTensorImpl();
    int64_t offset = flat.storage_offset();
    auto** items = PySequence_Fast_ITEMS(seq);
    for (size_t i = 0; i < shapes_.size(); ++i) {
      if (Py_TYPE(items[i]) != reinterpret_cast<PyTypeObject*>(THPVariableClass)) return false;
      const auto& g = THPVariable_Unpack(items[i]);
      if (!g.is_view() || g._base().unsafeGetTensorImpl() != root ||
          g.device() != flat.device() || g.scalar_type() != flat.scalar_type() || g.requires_grad() ||
          g.is_neg() || g.is_conj() || g.layout() != c10::kStrided || g.is_nested() ||
          g.unsafeGetTensorImpl()->has_symbolic_sizes_strides() ||
          !g.sizes().equals(shapes_[i]) || !g.is_contiguous() || g.storage_offset() != offset ||
          offset > std::numeric_limits<int64_t>::max() - numels_[i]) return false;
      // A .set_ on a view can change its storage independently of the base.
      // Defer that unusual case to the exact original Python implementation.
      if (g.storage().unsafeGetStorageImpl() != flat.storage().unsafeGetStorageImpl()) return false;
      offset += numels_[i];
    }
    return true;
  }
 private:
  std::vector<Dims> shapes_;
  Dims numels_;
};
}

PYBIND11_MODULE(_hstu_host_guard, m) {
  py::class_<Plan>(m, "Plan")
      .def(py::init<std::vector<Dims>, Dims>())
      .def("matches", &Plan::matches);
}
