"""Patch only the private non-compat context-presence cache in the pinned PTA TU."""
import hashlib
from pathlib import Path


def replace(src,old,new):
    assert src.count(old)==1,old
    return src.replace(old,new)


def patch(src):
    assert hashlib.sha256(src.encode()).hexdigest()=="d1169bd6f1f43a15456f9a5fda96b1bbeeb31ad97ec1529eb5daa20d8b580616"
    src=replace(src,"#include <mutex>","#include <mutex>\n#include <atomic>\n#include <cstdlib>\n#include <cstring>")
    anchor="std::recursive_mutex mtx;"
    src=replace(src,anchor,anchor+"\n"+Path(__file__).with_name("positive.inc").read_text())
    old="""aclError ResetUsedDevices()
{
    std::lock_guard<std::recursive_mutex> lock(mtx);"""
    src=replace(src,old,old+"\n    ActiveContextReset reset;")
    old="""    if (at_npu::native::env::CheckCompatibleImpl()) {
        return hasPrimaryContext(device);
    }
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if (used_devices.find(device) == used_devices.end()) {
        return false;
    }
    return used_devices[device] != nullptr;"""
    new="""    if (at_npu::native::env::CheckCompatibleImpl()) {
        count_context_read(3);
        return hasPrimaryContext(device);
    }
    const uint64_t bit = context_bit(device);
    if (bit && (active_context_mask.load(std::memory_order_acquire) & bit)) {
        count_context_read(0);
        return true;
    }
    std::lock_guard<std::recursive_mutex> lock(mtx);
    count_context_read(1);
    if (used_devices.find(device) == used_devices.end()) {
        return false;
    }
    const bool present = used_devices[device] != nullptr;
    if (present && bit && context_reset_depth == 0) {
        active_context_mask.fetch_or(bit, std::memory_order_release);
        count_context_read(2);
    }
    return present;"""
    return replace(src,old,new)
