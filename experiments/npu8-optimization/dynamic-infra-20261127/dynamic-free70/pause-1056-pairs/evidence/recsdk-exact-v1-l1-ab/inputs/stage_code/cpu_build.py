"""Continue an isolated 9.1.0 build without an NPU; package only the dense experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SOC="ascend910_93"
OFFICIAL=Path("/usr/local/Ascend/cann-9.1.0/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/ascendc_compile_kernel.py")


def call(args,cwd=None):
    print("+",*args,flush=True)
    subprocess.run(args,cwd=cwd,check=True)


def make_configs(build):
    mk,=build.glob("op_kernel/CMakeFiles/*_gen_ops_config.dir/build.make")
    tgt=mk.relative_to(build).parent.with_suffix("")
    call(["make","-f",str(mk),str(tgt)],cwd=build)


def stage(src,build,op,out):
    call(["cmake","--build",str(build),"--target","package","-j","2"])
    work=src/"cust_op/ascendc_op/ai_core_op"/op/"v220"
    call(["bash",str(src/"cust_op/scripts/extract_custom_opp_runs.sh"),op,str(out)],cwd=work)
    path=out/"vendors"/op/"op_api/lib/libcust_opapi.so"
    assert path.is_file(),path


def backward_target(build):
    tgt="HstuDenseBackward_"+SOC
    mk=build/f"op_kernel/CMakeFiles/{tgt}.dir/build.make"
    txt=mk.read_text()
    needle="python3 "+str(OFFICIAL)
    assert txt.count(needle)==1
    mk.write_text(txt.replace(needle,"python3 /recsdk-tools/cpu_build.py compile-bf16"))
    call(["cmake","--build",str(build),"--target",tgt,"-j","2"])
    make_configs(build)


def compile_bf16():
    assert hashlib.sha256(OFFICIAL.read_bytes()).hexdigest()=="429b9c301169b2cca1a48e991661dfd5755cf7f2df05da3926fe3a5de7fdb926"
    sys.path.insert(0,str(OFFICIAL.parent))
    from ascendc_compile_kernel import CompileKernel,args_parse
    del sys.argv[1]
    args=args_parse()
    assert args.op_type=="HstuDenseBackward" and args.enable_binary!="False"
    c=CompileKernel(args)
    c.clean()
    c.ascendc_gen_impl()
    c.ascendc_gen_param()
    c.ascendc_put_json()
    c.ascendc_put_tiling()
    selected=[]
    for sh in c.compile_sh:
        param=Path(re.search(r"--input_param=(\S+)",Path(sh).read_text()).group(1))
        assert param.resolve().is_relative_to(Path(c.working_dir).resolve())
        data=json.loads(param.read_text())
        if all(row["inputs"][0]["dtype"]=="bfloat16" for row in data["op_list"]):
            selected.append(sh)
            print("Selected official BF16 variant",param.name,flush=True)
    assert selected
    c.compile_sh=selected
    c.ascendc_build()
    c.clean()


def main(build):
    assert not list(Path("/dev").glob("davinci*")) and not Path("/dev/devmm_svm").exists()
    assert os.getenv("RECSDK_OP_SKIP_SYS_INSTALL")=="1"
    src,=build.glob("RecSDK-cacb4b2bab4fb0170fb9c5254498d0469e5bba60")
    ops=src/"cust_op/ascendc_op/ai_core_op"
    forward="hstu_dense_forward"
    backward="hstu_dense_backward"
    fbuild=ops/forward/"v220"/forward/"build_out"
    binaries=fbuild/"op_kernel/ascendc_kernels/binary"/SOC
    bf16=binaries/forward/"HstuDenseForward_859c4c0dbbf43b6274c8224262ec21ac.o"
    assert bf16.read_bytes().startswith(b"\x7fELF") and bf16.with_suffix(".json").is_file()
    out=build/"opp"
    if not (out/"vendors"/forward/"op_api/lib/libcust_opapi.so").exists():
        backups=build/"unrequested-kernel-products"
        backups.mkdir(exist_ok=True)
        for op in ("hstu_jagged_forward","hstu_paged_forward"):
            if (binaries/op).exists():
                (binaries/op).rename(backups/op)
        make_configs(fbuild)
        stage(src,fbuild,forward,out)
    template=ops/"custom_op_template/build.sh"
    original='cmake --build "$BUILD_DIR" --target binary -j${RECSDK_OP_BUILD_JOBS}'
    cmd='python3 /recsdk-tools/cpu_build.py backward-target "$BUILD_DIR"'
    t=template.read_text()
    if cmd not in t:
        assert t.count(original)==1
        template.write_text(t.replace(original,cmd))
    if not (out/"vendors"/backward/"op_api/lib/libcust_opapi.so").exists():
        bwork=ops/backward/"v220"
        call(["bash","run.sh","--ai-core","ai_core-Ascend910_93"],cwd=bwork)
        call(["bash",str(src/"cust_op/scripts/extract_custom_opp_runs.sh"),backward,str(out)],cwd=bwork)
    adapter=src/"cust_op/framework/torch_plugin/torch_library/hstu"
    # CMake reads the PyTorch C++ ABI from torch._C. No device plugin is needed in the CPU-only builder.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"]="0"
    call(["cmake","-S",str(adapter),"-B",str(build/"adapter"),"-DBUILD_VER=v220"])
    call(["cmake","--build",str(build/"adapter"),"-j","2"])
    libs=[build/"adapter/libhstu_dense_ops.so",*(out/"vendors"/v/"op_api/lib/libcust_opapi.so" for v in (forward,backward))]
    (build/"libraries.sha256").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in libs))
    print("CPU_BUILD_COMPLETE", flush=True)


if __name__=="__main__":
    if sys.argv[1]=="compile-bf16":
        compile_bf16()
    elif sys.argv[1]=="backward-target":
        backward_target(Path(sys.argv[2]).resolve(strict=True))
    else:
        main(Path(sys.argv[1]).resolve(strict=True))
