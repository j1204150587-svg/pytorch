} // namespace c10_npu

int main()
{
    using namespace c10_npu;
    const bool enabled=std::string(std::getenv("NPU_CONTEXT_POSITIVE_MASK"))=="1";
    assert(!isDeviceCtxActive(0) && !isDeviceCtxActive(64) && !isDeviceCtxActive(-1));
    assert(MaybeSetDevice(2)==0 && targetDeviceIndex==2);
    std::cout<<"absent states and lazy target\n";
    fail_set=true;
    assert(SetDevice(0)!=0 && !isDeviceCtxActive(0)); fail_set=false;
    fail_context=true;
    try {SetDevice(0);assert(false);} catch(const std::runtime_error&) {}
    assert(!isDeviceCtxActive(0)); fail_context=false;
    assert(ResetUsedDevices()==0);
    local_device=-1;
    null_context=true; assert(SetDevice(0)==0 && !isDeviceCtxActive(0)); null_context=false;
    assert(ResetUsedDevices()==0); local_device=-1;
    std::cout<<"SetDevice failure, context error and explicit null never publish positive\n";
    assert(SetDevice(0)==0); assert(isDeviceCtxActive(0));
    uint64_t h=npu_active_context_counter(0);
    assert(isDeviceCtxActive(0) && npu_active_context_counter(0)-h==uint64_t(enabled));
    LazySetDevice(63); assert(isDeviceCtxActive(63) && isDeviceCtxActive(63));
    LazySetDevice(64); assert(isDeviceCtxActive(64) && isDeviceCtxActive(64));
    assert(isDeviceCtxActive(256) && isDeviceCtxActive(-256));
    assert(!isDeviceCtxActive(65)); // no change to original int8-key conversion on table fallback
    std::cout<<"0 and 63 cacheable, 64, negative, above-range preserve original table semantics\n";
    at_npu::native::env::compatible=true;
    primary_active=false; assert(!isDeviceCtxActive(0));
    primary_active=true; assert(isDeviceCtxActive(0));
    try {isDeviceCtxActive(-1);assert(false);}catch(const std::runtime_error&) {}
    assert(primary_calls==2);
    at_npu::native::env::compatible=false;
    std::cout<<"compatibility re-queries primary context, including invalid-device error\n";
    // Fail mid-reset: the unmodified used_devices map remains authoritative even if a mock
    // device has already been reset. An external reader blocks rather than observing stale bits.
    std::promise<void> entered, release;
    std::shared_future<void> may_exit=release.get_future().share();
    bool callback_first=true;
    reset_callback=[&] {
        if(callback_first) {callback_first=false;assert(isDeviceCtxActive(0));entered.set_value();may_exit.wait();}
    };
    auto resetting=std::async(std::launch::async,[]{return ResetUsedDevices();});
    entered.get_future().wait();
    std::promise<void> launched;
    auto reading=std::async(std::launch::async,[&]{launched.set_value();return isDeviceCtxActive(0);});
    launched.get_future().wait();
    assert(reading.wait_for(std::chrono::milliseconds(40))==std::future_status::timeout);
    release.set_value();
    assert(resetting.get()==0 && !reading.get() && !isDeviceCtxActive(0));
    reset_callback={};
    std::cout<<"successful reset with recursive lookup and concurrent reader\n";
    local_device=-1;LazySetDevice(0);LazySetDevice(1);
    assert(isDeviceCtxActive(0) && isDeviceCtxActive(1));
    unsigned calls=0;
    reset_callback=[&] { if(++calls==2) fail_reset=true; };
    assert(ResetUsedDevices()!=0); fail_reset=false;reset_callback={};
    assert(active_context_mask.load()==0 && isDeviceCtxActive(0) && isDeviceCtxActive(1));
    throw_reset=true;
    try {ResetUsedDevices();assert(false);}catch(const std::runtime_error&){}
    throw_reset=false;
    assert(context_reset_depth==0 && active_context_mask.load()==0 && isDeviceCtxActive(0));
    assert(ResetUsedDevices()==0 && !isDeviceCtxActive(0));
    std::cout<<"partial reset failure and unwinding retain original table and refill under lock\n";
    return 0;
}
