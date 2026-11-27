#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <exception>
#include "torch_npu/csrc/inductor/aoti_package/shape_handling.h"

static PyObject* install(PyObject*, PyObject* extension) {
  if (!PyModule_Check(extension)) {
    PyErr_SetString(PyExc_TypeError, "expected torch_npu._C");
    return nullptr;
  }
  try {
    THNPShapeHandling_init(extension);
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
  Py_RETURN_NONE;
}
static PyMethodDef methods[] = {{"install", install, METH_O, nullptr}, {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_pr42634_shape", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__pr42634_shape() { return PyModule_Create(&module); }
