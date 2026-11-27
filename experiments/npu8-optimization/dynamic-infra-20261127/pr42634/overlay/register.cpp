#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <exception>
#include "torch_npu/_inductor/experimental/python_wrapper_fast_launch/csrc/bindings.h"

// The PR's registration routine receives exactly torch_npu._C, as it does
// from InitNpuBindings.cpp in the upstream build. Its source remains verbatim.
static PyObject* install(PyObject*, PyObject* extension) {
  if (!PyModule_Check(extension)) {
    PyErr_SetString(PyExc_TypeError, "expected torch_npu._C");
    return nullptr;
  }
  try {
    RegisterNPUFastLaunchBindings(extension);
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {{"install", install, METH_O, nullptr}, {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_pr42634_register", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__pr42634_register() { return PyModule_Create(&module); }
