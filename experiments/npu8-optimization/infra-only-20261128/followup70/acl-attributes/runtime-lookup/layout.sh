#!/usr/bin/env bash
# Read-only capture against the training container's files and the previously built CANN source.
set -eu
study=/mnt/paas/recsys-gr-ds019-host88-archive/dispatch-study-20261119-npu8/infra-only-followup70/analysis/cann-runtime-host
pid=$(docker inspect -f '{{.State.Pid}}' recsys-gr-ds019-npu8)
lib=/proc/$pid/root/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64
which readelf objdump nm
readelf --version |head -1
for f in libnnopbase.so libacl_rt.so libacl_rt_impl.so libruntime.so libruntime_v100.so; do
    sha256sum "$lib/$f"
done
grep -n -m1 ACL_STREAM_ATTR_CACHE_OP_INFO "/proc/$pid/root/usr/local/Ascend/cann-9.1.0/include/acl/acl_rt.h"
cd "$study/source"
sha256sum src/runtime/core/src/api_impl/{api_error.cc,api_impl.cc} src/runtime/core/{src/runtime.cc,inc/runtime.hpp} src/runtime/api/api_c_stream.cc
sed -n '444,453p;490,501p' src/runtime/api/api_c_stream.cc
sed -n '5915,5929p' src/runtime/core/src/api_impl/api_error.cc
sed -n '7781,7798p' src/runtime/core/src/api_impl/api_impl.cc
sed -n '187,197p' src/runtime/core/inc/runtime.hpp
sed -n '5409,5420p' src/runtime/core/src/runtime.cc
cd "$study/build/src/runtime/CMakeFiles/runtime_v100.dir/core/src"
for o in api_impl/api_error.cc.o api_impl/api_impl.cc.o; do
    readelf -Wr -C "$o" | grep -E 'GetStreamCacheOpInfoSwitch'
done
objdump -drC -j .text --start-address=0x2ba80 --stop-address=0x2bb08 api_impl/api_error.cc.o
objdump -drC -j .text --start-address=0x184 --stop-address=0x194 api_impl/api_impl.cc.o
cd "$lib"
readelf -Wr libruntime_v100.so |grep -E ' 59f930$| 674770$| 59fe50$| 675828$|^00000000006767c0|^0000000000675708|^000000000067fdf8'
strings -tx libruntime_v100.so |grep -E ' N3cce7runtime(7ApiImpl|17ApiErrorDecorator)E$'
for range in '0x1b6d90 0x1b6dd0' '0x303630 0x303654'; do
    set -- $range
    objdump -d --start-address=$1 --stop-address=$2 libnnopbase.so
done
objdump -d --start-address=0x50b60 --stop-address=0x50b94 libacl_rt_impl.so
for range in '0x1079b8 0x107a3c' '0x2550e0 0x255154' '0x61bd0 0x61be0'; do
    set -- $range
    objdump -d --start-address=$1 --stop-address=$2 libruntime_v100.so
done
