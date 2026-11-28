"""Write the runner recipe after the private CANN kernel passes its full primitive gate."""
import argparse
import json
from pathlib import Path, PurePosixPath

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("container_build",type=PurePosixPath)
p.add_argument("result",type=Path,help="downloaded private whole-kernel npu.json")
p.add_argument("recipe",type=Path)
a=p.parse_args()
check=json.loads(a.result.read_text())
assert check["mode"]=="exact_native" and check["complete"] and len(check["cases"])==66
assert all(len(c["differences"])==4 and not any(v["differing_bf16_words"] for v in c["differences"].values()) for c in check["cases"])
assert a.container_build.is_absolute()
d=a.container_build
opp=d/"opp/vendors"
recipe=dict(stage="recsdk_dense_exact",
            common={"NPU_RECSDK_LIBRARY":str(d/"adapter/libhstu_dense_ops.so"),
                    "ASCEND_CUSTOM_OPP_PATH":":".join(str(opp/name) for name in ("hstu_dense_forward","hstu_dense_backward"))},
            A={"NPU_RECSDK_EAGER_ORDER":"0"},B={"NPU_RECSDK_EAGER_ORDER":"1"},
            activation=[{"file":"recsdk_exact.json","A":{"passed":True,"num_blocks":4,"mode":"0"},"B":{"passed":True,"num_blocks":4,"mode":"1"}}])
a.recipe.write_text(json.dumps(recipe,indent=2)+"\n")
