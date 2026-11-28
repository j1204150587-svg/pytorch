"""Export exactly the built source delta and verify patch application against pristine source hashes."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=="__main__":
    here=Path(__file__).resolve().parent
    r=json.loads((here/"build/build-receipt.json").read_bytes())
    def path(raw):
        p=Path(raw)
        if p.exists():
            return p
        return here.parents[1]/p.relative_to("/artifacts")
    plugin=path(r["base"])/"checkout/third_party/op-plugin"
    stage=path(r["stage"])
    patch=here/"from-stock.patch"
    lines=[]
    with tempfile.TemporaryDirectory() as tmp:
        tree=Path(tmp)
        for name,want in r["patched_sources_sha256"].items():
            target=stage/name
            if sha(target)!=want:
                raise ValueError(name)
            src=plugin/name
            if name in r["original_sources_sha256"]:
                if sha(src)!=r["original_sources_sha256"][name]:
                    raise ValueError(name)
                dst=tree/name
                dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(src,dst)
                before=src.read_text().splitlines(True)
                old="a/"+name
            else:
                if src.exists():
                    raise ValueError(name)
                old="/dev/null"
                before=[]
            lines.extend(difflib.unified_diff(before,target.read_text().splitlines(True),fromfile=old,tofile="b/"+name))
        patch.write_text("".join(lines),encoding="utf-8")
        cmd=["git","apply","--no-index","--whitespace=nowarn"]
        subprocess.run([*cmd,"--check",str(patch)],cwd=tree,check=True)
        subprocess.run([*cmd,str(patch)],cwd=tree,check=True)
        if any(sha(tree/name)!=v for name,v in r["patched_sources_sha256"].items()):
            raise ValueError("Patched tree differs from build receipt")
    (here/"patch-verification.json").write_text(json.dumps({"patch_sha256":sha(patch),"library_sha256":r["library_sha256"],
        "base":r["op_plugin_sha"],"applied_sources_sha256":r["patched_sources_sha256"]},indent=2)+"\n")
