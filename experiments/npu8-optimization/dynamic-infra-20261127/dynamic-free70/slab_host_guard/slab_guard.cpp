#define PY_SSIZE_T_CLEAN
#include <Python.h>

namespace {
struct Ref {
  explicit Ref(PyObject* value) : value(value) {}
  ~Ref() { Py_XDECREF(value); }
  Ref(const Ref&) = delete;
  PyObject* value;
};

const char* names[] = {"weight", "data_ptr", "requires_grad", "grad",
                       "_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
                       "_backward_pre_hooks", "_backward_hooks", "_post_accumulate_grad_hooks"};
constexpr int num_names = sizeof(names) / sizeof(names[0]);
struct State { PyObject* attrs[num_names]; };

// Use the live Python attribute, truth and iteration protocols. This preserves
// table/weight replacement and per-object hooks without caching a prior answer.
PyObject* matches(PyObject* self, PyObject* args) {
  PyObject *live, *tables, *weights, *pointers;
  if (!PyArg_ParseTuple(args, "OOOO:matches", &live, &tables, &weights, &pointers))
    return nullptr;
  if (!PyTuple_CheckExact(tables) || !PyTuple_CheckExact(weights) || !PyTuple_CheckExact(pointers)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple snapshots");
    return nullptr;
  }
  Py_ssize_t n = PyObject_Length(live);
  if (n < 0) return nullptr;
  if (n != PyTuple_GET_SIZE(tables)) Py_RETURN_FALSE;
  // zip() consumes at most the shortest iterable. The constructor creates
  // equal lengths; altered snapshots use the public Python implementation.
  if (n != PyTuple_GET_SIZE(weights) || n != PyTuple_GET_SIZE(pointers)) {
    PyErr_SetString(PyExc_ValueError, "snapshot lengths differ");
    return nullptr;
  }
  State* state = static_cast<State*>(PyModule_GetState(self));
  Ref iterator(PyObject_GetIter(live));
  if (!iterator.value) return nullptr;
  for (Py_ssize_t i = 0; i < n; ++i) {
    Ref table(PyIter_Next(iterator.value));
    if (!table.value) {
      if (PyErr_Occurred()) return nullptr;
      Py_RETURN_TRUE;
    }
    if (table.value != PyTuple_GET_ITEM(tables, i)) Py_RETURN_FALSE;
    PyObject* param = PyTuple_GET_ITEM(weights, i);
    Ref actual(PyObject_GetAttr(table.value, state->attrs[0]));
    if (!actual.value) return nullptr;
    if (actual.value != param) Py_RETURN_FALSE;
    Ref ptr(PyObject_CallMethodNoArgs(param, state->attrs[1]));
    if (!ptr.value) return nullptr;
    int changed = PyObject_RichCompareBool(ptr.value, PyTuple_GET_ITEM(pointers, i), Py_NE);
    if (changed < 0) return nullptr;
    if (changed) Py_RETURN_FALSE;
    Ref trainable(PyObject_GetAttr(param, state->attrs[2]));
    if (!trainable.value) return nullptr;
    int enabled = PyObject_IsTrue(trainable.value);
    if (enabled < 0) return nullptr;
    if (!enabled) Py_RETURN_FALSE;
    Ref grad(PyObject_GetAttr(param, state->attrs[3]));
    if (!grad.value) return nullptr;
    if (grad.value != Py_None) Py_RETURN_FALSE;
    for (int j = 4; j < num_names; ++j) {
      Ref hooks(PyObject_GetAttr(j < 8 ? table.value : param, state->attrs[j]));
      if (!hooks.value) return nullptr;
      int present = PyObject_IsTrue(hooks.value);
      if (present < 0) return nullptr;
      if (present) Py_RETURN_FALSE;
    }
  }
  Py_RETURN_TRUE;
}

int visit_state(PyObject* mod, visitproc visit, void* arg) {
  State* state = static_cast<State*>(PyModule_GetState(mod));
  for (auto attr : state->attrs) Py_VISIT(attr);
  return 0;
}
int clear_state(PyObject* mod) {
  State* state = static_cast<State*>(PyModule_GetState(mod));
  for (auto& attr : state->attrs) Py_CLEAR(attr);
  return 0;
}
PyMethodDef methods[] = {{"matches", matches, METH_VARARGS, "Evaluate a fresh per-table live guard."},
                         {nullptr, nullptr, 0, nullptr}};
PyModuleDef definition = {PyModuleDef_HEAD_INIT, "_hstu_slab_guard", nullptr, sizeof(State), methods,
                          nullptr, visit_state, clear_state, nullptr};
}
PyMODINIT_FUNC PyInit__hstu_slab_guard() {
  PyObject* module = PyModule_Create(&definition);
  if (!module) return nullptr;
  State* state = static_cast<State*>(PyModule_GetState(module));
  for (int i = 0; i < num_names; ++i) {
    state->attrs[i] = PyUnicode_InternFromString(names[i]);
    if (!state->attrs[i]) {
      Py_DECREF(module);
      return nullptr;
    }
  }
  return module;
}
