#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <future>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

using aclError = int;
using aclrtContext = void*;
constexpr aclError ACL_ERROR_NONE=0;
namespace c10 { using DeviceIndex=int8_t; }
namespace at_npu::native::env {
bool compatible=false;
bool CheckCompatibleImpl() { return compatible; }
}
#define TORCH_CHECK(cond,...) do { if(!(cond)) throw std::runtime_error("framework check"); } while(0)
#define NPU_CHECK_ERROR_WITHOUT_UCE(expr) do { if((expr)!=0) throw std::runtime_error("acl error"); } while(0)
#define ASCEND_LOGI(...) ((void)0)
namespace c10_npu {
thread_local int local_device=-1;
thread_local int targetDeviceIndex=-1;
std::unordered_map<int8_t,aclrtContext> used_devices;
std::recursive_mutex mtx;
bool NeedMainThreadBind(){return false;}
void SetThreadAffinity(int){}
int device_count(){return 4;}
bool primary_active=false, fail_set=false, fail_context=false, null_context=false, fail_reset=false, throw_reset=false;
int primary_calls=0, reset_calls=0;
int aclrtSetDevice(c10::DeviceIndex) {return fail_set?1:0;}
int aclrtGetCurrentContext(aclrtContext* out)
{
    if(fail_context) return 1;
    *out=null_context?nullptr:reinterpret_cast<void*>(uintptr_t{1});
    return 0;
}
namespace acl {
int AclrtGetPrimaryCtxState(c10::DeviceIndex,void*,int32_t* value)
{
    ++primary_calls;
    *value=primary_active;
    return 0;
}
}
std::function<void()> reset_callback;
int aclrtResetDevice(int8_t) {
    ++reset_calls;
    if(reset_callback)reset_callback();
    if(throw_reset)throw std::runtime_error("mock vendor exception");
    return fail_reset?1:0;
}
