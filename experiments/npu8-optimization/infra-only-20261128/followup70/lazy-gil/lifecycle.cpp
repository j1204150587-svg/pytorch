// Exercise the candidate's runtime guard across actual CPython interpreter teardown.
#include <Python.h>
#include "config.inc"
#include <cassert>

static unsigned final_callbacks = 0;
void after_teardown()
{
    ++final_callbacks;
    assert(!pythonRuntimeLive());
}

int main()
{
    assert(!pythonRuntimeLive());
    for (unsigned pass = 0; pass < 2; ++pass) {
        Py_Initialize();
        assert(pythonRuntimeLive());
        assert(Py_AtExit(after_teardown) == 0);
        assert(Py_FinalizeEx() == 0);
        assert(!pythonRuntimeLive());
    }
    assert(final_callbacks == 2);
    std::puts("{\"fresh_and_restart\":2,\"after_teardown_callbacks\":2}");
}
