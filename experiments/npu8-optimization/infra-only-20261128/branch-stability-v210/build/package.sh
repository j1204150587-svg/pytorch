#!/bin/bash
set -euo pipefail
out=/data/recsys-gr-ds019-host88/artifacts/infra-only-20261128/branch-stability-v210/build
inner=/artifacts/infra-only-20261128/branch-stability-v210/build
image=sha256:c342a038f6476c794fb364d96479286ebd456021e86dba0d3b715ce4cb983e33
test "$(cat "$out"/build.exit)" = 0
pid=$(docker inspect -f '{{.State.Pid}}' recsys-gr-ds019-npu8)
merged=$(docker inspect -f '{{.GraphDriver.Data.MergedDir}}' recsys-gr-ds019-npu8)
pkg=/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu
stock="$merged$pkg"
test "$(stat -Lc '%D:%i' "$stock")" = "$(stat -Lc '%D:%i' "/proc/$pid/root$pkg")"
docker run --name recsys-dynbranch-e3cda63-package2 --runtime runc --network none --cpus 3 --memory 6g --entrypoint /bin/bash -v "$out:$inner" -v "$stock:/training-stock:ro" -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -e "BRANCH_OUTPUT=$inner" -e PYTHONDONTWRITEBYTECODE=1 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$image" -ec 'python -B "$BRANCH_OUTPUT/package_branch.py"; PYTHONPATH="$BRANCH_OUTPUT/overlay" python -B "$BRANCH_OUTPUT/verify_overlay.py"'
