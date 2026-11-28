#!/bin/bash
set -euo pipefail
out=/data/recsys-gr-ds019-host88/artifacts/infra-only-20261128/branch-stability-v210/build
image=sha256:c342a038f6476c794fb364d96479286ebd456021e86dba0d3b715ce4cb983e33
docker run --rm --read-only --runtime runc --network none --cpus 0.5 --memory 512m --entrypoint /bin/bash -v "$out:/proof:ro" -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro "$image" -ec 'env -u ACLNN_CACHE_LIMIT python -B /proof/cache_limit.py; env ACLNN_CACHE_LIMIT=65536 python -B /proof/cache_limit.py'
pid=$(docker inspect -f '{{.State.Pid}}' recsys-gr-ds019-npu8)
lib=/proc/$pid/root/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libnnopbase.so
sha256sum "$lib"
nm -DSC "$lib" | grep -E 'OpExecCacheManager::(GetCacheSizeLimit|OpExecCacheManager)\('
objdump -dC --start-address=0x175ec0 --stop-address=0x1760b0 "$lib"
objdump -dC --start-address=0x178a70 --stop-address=0x178aa0 "$lib"
objdump -s --start-address=0x367660 --stop-address=0x367678 "$lib"
