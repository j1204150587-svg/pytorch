#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
core=/artifacts/pta-build-UK5IES/checkout
cd "$core"
test ! -e /dev/davinci_manager && test ! -e /dev/davinci0
exec > >(tee /result/configure-build.log) 2>&1
date -Is
git log -1 --format='%H %P %s'
git -C third_party/op-plugin rev-parse HEAD
cmake -S "$core" -B "$core/build"
python - <<'PY'
import json
from pathlib import Path
root=Path("/artifacts/pta-build-UK5IES/checkout")
commands=json.loads((root/"build/compile_commands.json").read_bytes())
names=["op_api_common.cpp","StructKernelNpuOpApi.cpp","EmbeddingKernelNpuOpApi.cpp","NPUQueue.cpp","OpCommand.cpp"]
groups={name:[c for c in commands if Path(c['file']).name==name] for name in names}
for name, rows in groups.items():
    assert len(rows)==1,(name,len(rows))
    if name in names[:3]:
        assert Path(rows[0]['file']).parent==root/"build/embedding",(name,rows)
print("CMake targets verified",[(n,rows[0]["file"]) for n,rows in groups.items()],flush=True)
Path("/result/branch-tu-commands.json").write_text(json.dumps(groups,indent=2))
PY
cmake --build "$core/build" --target torch_npu -j8 -v > /result/compile-link.log 2>&1
test -s "$core/build/packages/torch_npu/lib/libtorch_npu.so"
cp -p "$core/build/packages/torch_npu/lib/libtorch_npu.so" /result/libtorch_npu.so
sha256sum /result/libtorch_npu.so
date -Is
