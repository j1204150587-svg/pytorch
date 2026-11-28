#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <c10/util/strides.h>
#include <torch/csrc/autograd/python_variable.h>

namespace {
bool plain(const at::Tensor& tensor) {
  return (tensor.device().is_cpu() || tensor.device().is_privateuseone()) &&
      tensor.layout() == at::kStrided && tensor.scalar_type() == at::kBFloat16 &&
      !tensor.requires_grad() && !tensor.is_inference() && !tensor.has_names() &&
      !tensor.is_nested() && !tensor.is_conj() && !tensor.is_neg() &&
      !tensor.unsafeGetTensorImpl()->has_symbolic_sizes_strides() &&
      !at::isTensorSubclassLike(tensor) && !tensor._fw_grad(0).defined() &&
      !(tensor.is_view() && tensor._base().requires_grad());
}

bool exact_tensor(PyObject* obj) {
  return Py_TYPE(obj) == reinterpret_cast<PyTypeObject*>(THPVariableClass);
}

struct Slot {
  Py_ssize_t id;
  at::Tensor grad;
  at::Tensor view;
};

class Prepared {
 public:
  explicit Prepared(py::list flags) : flags_(std::move(flags)) {}

  void publish() {
    if (consumed_) throw std::logic_error("IPG views already published");
    consumed_ = true;
    // GIL remains held; flags are an exact built-in list of bool. set_data()
    // updates each *original* gradient's TensorImpl, including its live aliases.
    for (const auto& slot : slots_) {
      Py_INCREF(Py_True);
      if (PyList_SetItem(flags_.ptr(), slot.id, Py_True) != 0) throw py::error_already_set();
      slot.grad.set_data(slot.view.variable_data());
    }
  }

  static std::shared_ptr<Prepared> make(py::handle target, py::handle group, py::handle flags) {
    if (!exact_tensor(target.ptr()) || !PyList_CheckExact(group.ptr()) ||
        !PyList_CheckExact(flags.ptr())) return nullptr;
    const auto& dst = THPVariable_Unpack(target.ptr());
    if (!plain(dst) || dst.dim() != 1 || dst.stride(0) != 1 || dst.storage_offset() != 0 ||
        dst.is_view()) return nullptr;
    auto out = std::make_shared<Prepared>(py::reinterpret_borrow<py::list>(flags));
    auto& slots = out->slots_;
    const auto count = PyList_GET_SIZE(group.ptr());
    slots.reserve(count);
    int64_t offset = 0;
    for (Py_ssize_t i = 0; i < count; ++i) {
      auto* row = PyList_GET_ITEM(group.ptr(), i);
      if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 4) return nullptr;
      auto* id = PyTuple_GET_ITEM(row, 0);
      auto* grad = PyTuple_GET_ITEM(row, 1);
      if (!PyLong_CheckExact(id) || !exact_tensor(grad)) return nullptr;
      auto idx = PyLong_AsSsize_t(id);
      if (idx == -1 && PyErr_Occurred()) throw py::error_already_set();
      if (idx < 0 || idx >= PyList_GET_SIZE(flags.ptr()) ||
          PyList_GET_ITEM(flags.ptr(), idx) != Py_False) return nullptr;
      const auto& g = THPVariable_Unpack(grad);
      if (!plain(g) || g.device() != dst.device() || !g.is_contiguous() ||
          g.numel() <= 0 || g.numel() > dst.numel()-offset) return nullptr;
      // The preceding IPG guard has verified source adjacency, grad shape and
      // parameter identity. A single canonical view equals narrow(...).view(shape).
      auto v = dst.as_strided(g.sizes(), c10::contiguous_strides(g.sizes()), offset);
      slots.push_back({idx, g, std::move(v)});
      offset += g.numel();
    }
    if (count < 2 || offset != dst.numel()) return nullptr;
    return out;
  }

 private:
  py::list flags_;
  std::vector<Slot> slots_;
  bool consumed_ = false;
};
}

PYBIND11_MODULE(_hstu_ipg_host_views, m) {
  py::class_<Prepared, std::shared_ptr<Prepared>>(m, "Prepared")
      .def("publish", &Prepared::publish);
  m.def("prepare", &Prepared::make);
}
