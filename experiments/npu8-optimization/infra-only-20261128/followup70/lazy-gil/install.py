"""Publish a successful completed Python lazy init for native hot-path callers."""
from pathlib import Path
import sys

SOURCE = "torch_npu/csrc/utils/LazyInit.cpp"


def apply(source):
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError("unexpected source: " + old[:60])
        source = source.replace(old,new)
    replace('namespace torch_npu {',Path(__file__).with_name('config.inc').read_text()+'\nnamespace torch_npu {')
    replace("    pybind11::gil_scoped_acquire g;",'''    const bool audit = lazyAudit();
    const bool fast = lazyEnabled();
    if (audit) countLazy(lazyCalls);
    if (fast && pythonReady.load(std::memory_order_acquire) && pythonRuntimeLive()) {
        if (audit) countLazy(lazyHits);
        return;
    }
    const auto start = audit ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    pybind11::gil_scoped_acquire g;
    if (audit) {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count();
        lazyWaitNs.fetch_add(ns, std::memory_order_relaxed);
        uint64_t old = lazyMaxNs.load(std::memory_order_relaxed);
        while (old < static_cast<uint64_t>(ns) && !lazyMaxNs.compare_exchange_weak(
            old, ns, std::memory_order_relaxed)) {}
    }''')
    replace('        npu_run_yet = true;\n    }','''        npu_run_yet = true;
    }
    // A reentrant _lazy_init can return before another Python caller has
    // completed its deferred callbacks. Only publish the committed state.
    if (fast && pythonRuntimeLive()) {
        PyObject* module = PyDict_GetItemString(PyImport_GetModuleDict(), "torch_npu.npu");
        if (module && PyModule_Check(module)) {
            PyObject* dict = PyModule_GetDict(module);
            if (PyDict_GetItemString(dict, "_initialized") == Py_True &&
                PyDict_GetItemString(dict, "_is_internal_in_bad_fork") == Py_False) {
                pythonReady.store(true, std::memory_order_release);
            }
        }
    }''')
    replace('    npu_run_yet = false;','''    pythonReady.store(false, std::memory_order_release);
    npu_run_yet = false;''')
    return source


if __name__=="__main__":
    f=Path(sys.argv[1])/SOURCE
    f.write_text(apply(f.read_text()))
