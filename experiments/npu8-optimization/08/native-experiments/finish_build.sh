#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")"
out=$PWD
inner="/artifacts/${out##*/}"
build_pid=$(cat build.pid)
while [[ ! -s build.exit ]]; do
    if ! ps -p "$build_pid" -o stat= | grep -q '^[^Z]'; then
        echo "Build process is no longer live and has no terminal receipt"
        exit 1
    fi
    sleep 5
done
test "$(cat build.exit)" = 0

shopt -s nullglob
wheels=(checkout/dist/*.whl)
test "${#wheels[@]}" = 1
sha256sum "${wheels[0]}" > candidate-wheel.sha256
container=recsys-gr-ds019-npu8
docker exec "$container" python3 -m pip install --no-index --no-deps --no-compile \
    --target "$inner/package" "$inner/${wheels[0]}"
docker exec --env PYTHONPATH="$inner/package" --env PTA_EXPECTED_ROOT="$inner/package" \
    --workdir "$inner/work" "$container" bash -c '
        set -e
        source ./env.sh
        python3 -B -c "
import hashlib, json, os
from pathlib import Path
import torch
import torch_npu
p = Path(torch_npu.__file__).resolve()
root = Path(os.environ[\"PTA_EXPECTED_ROOT\"])
assert p.is_relative_to(root)
assert hasattr(torch_npu.npu.NPUGraph, \"_try_replay_with_copy\")
paths = {Path(line.split()[-1]) for line in Path(\"/proc/self/maps\").read_text().splitlines()
         if line.split()[-1].endswith(\"/libtorch_npu.so\")}
assert len(paths) == 1
library, = paths
assert library.is_relative_to(root)
with library.open(\"rb\") as source:
    digest = hashlib.file_digest(source, \"sha256\").hexdigest()
print(json.dumps(dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
                      module=str(p), library=str(library), library_sha256=digest)))
" ' > candidate-runtime.json
python3 -B run_suite.py gates gates-01 queued flat
