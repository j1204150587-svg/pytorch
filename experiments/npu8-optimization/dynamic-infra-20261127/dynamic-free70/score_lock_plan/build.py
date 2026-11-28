"""Add the verified lock resource to the actual physical-cap PR binding, in a separate overlay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite(text,needle,out):
    if text.count(needle)!=1:
        raise ValueError("Unexpected pinned bindings.cpp around "+needle[:65])
    return text.replace(needle,out)


def patch(src,here):
    src=rewrite(src,"#include <cstddef>","#include <atomic>\n#include <dlfcn.h>\n#include <cstddef>")
    src=rewrite(src,"struct FastLaunchPlan {",(here/"lock_resource.inc").read_text()+"\nstruct FastLaunchPlan {\n"
        +"  size_t syncLockOffset = 0;\n  std::shared_ptr<ScoreLockApi> scoreLockApi;")
    key="      offset = AlignOffset(offset, alignof(void*));\n      offset += sizeof(void*);"
    src=rewrite(src,key,key.replace("      offset +=", "      if (index == 0) plan.syncLockOffset = offset;\n      offset +="))
    key="  auto launchCall = [kernelStub = plan.kernelStub,"
    src=rewrite(src,key,key+"\n                     lockApi = plan.scoreLockApi, syncLockOffset = plan.syncLockOffset,")
    key="    void* args = packed.args.data();"
    src=rewrite(src,key,(here/"lock_submit.inc").read_text()+key)
    key="    return static_cast<int>(result);"
    src=rewrite(src,key,"    if (lockApi && result == RT_ERROR_NONE) lockApi->kernelReturns.fetch_add(1, std::memory_order_relaxed);\n"+key)
    key='  m.def("_npu_hstu_fast_launch_with_physical_cap", &FastLaunchWithPhysicalCap);'
    return rewrite(src,key,key+'''
  m.def("_npu_hstu_make_score_lock_plan", [](const std::string& name, py::object stub,
        const std::vector<std::string>& kinds, const std::string& npuUtilsPath) {
    TORCH_CHECK((name == "causal_score_forward" && kinds.size() == 3) ||
                (name == "causal_score_backward" && kinds.size() == 4), "unexpected score kernel");
    TORCH_CHECK(std::all_of(kinds.begin(), kinds.end(), [](auto& s) { return s == "tensor"; }),
                "score kernel expects tensor arguments");
    auto plan = MakeFastLaunchPlan(name, std::move(stub), kinds, false, 0, false, true);
    plan->scoreLockApi = std::make_shared<ScoreLockApi>(npuUtilsPath);
    return plan;
  });
  m.def("_npu_hstu_score_lock_counts", [](const std::shared_ptr<FastLaunchPlan>& plan) {
    TORCH_CHECK(plan && plan->scoreLockApi, "expected a score lock plan");
    auto& api = *plan->scoreLockApi;
    return py::make_tuple(api.allocations.load(), api.initialized.load(), api.kernelReturns.load());
  });
''')


def build(base,out):
    here=Path(__file__).resolve().parent
    metadata=json.loads((base/"build.json").read_bytes())
    if metadata["pr_binding_blob"]!="23290b574bb850929a14f279e5764e6e63595574":
        raise ValueError("wrong PR blob")
    argv=metadata["command"].copy()
    cpp,=[Path(v) for v in argv if v.endswith("/bindings.cpp")]
    if sha(cpp)!=metadata["patched_source_sha256"]:
        raise ValueError("physical-cap source differs from measured build")
    out.mkdir(parents=True,exist_ok=True)
    if (out/"package").exists():
        raise FileExistsError(out/"package")
    staged=Path(tempfile.mkdtemp(dir=out))
    target=staged/"bindings.cpp"
    target.write_text(patch(cpp.read_text(),here))
    argv[argv.index(str(cpp))]=str(target)
    index=argv.index("-o")+1
    name=Path(argv[index]).name
    binary=staged/name
    argv[index]=str(binary)
    subprocess.run(argv,check=True)
    pkg=staged/"package"
    def link(src,dst):
        try:
            os.link(src,dst)
        except OSError:
            shutil.copyfile(src,dst)
        return dst
    shutil.copytree(base/"package",pkg,symlinks=True,copy_function=link)
    basebin=base/"package"/"torch_npu"/name
    if sha(basebin)!=metadata["shared_library_sha256"]:
        raise ValueError("base overlay hash differs")
    os.replace(binary,pkg/"torch_npu"/name)
    pkg.rename(out/"package")
    (out/"build.json").write_text(json.dumps({"pr_binding_blob":metadata["pr_binding_blob"],
        "base_source":sha(cpp),"base_library":sha(basebin),"command":argv,
        "patched_source":sha(target),"shared_library":sha(out/"package"/"torch_npu"/name)},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("base",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    build(a.base,a.out)
