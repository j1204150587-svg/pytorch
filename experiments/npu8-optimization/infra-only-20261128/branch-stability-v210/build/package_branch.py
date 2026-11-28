"""Build an installable wheel from the branch ELF and the pinned image's unchanged wheel payload."""
from pathlib import Path
import base64
import csv
import hashlib
from importlib import metadata
import io
import json
import os
import shutil
import subprocess
import sys
from wheel.wheelfile import WheelFile
from zipfile import ZipFile

here=Path(__file__).resolve().parent
head="e3cda6399cb194c813530222478ea018f5aca5c9"
base="38e483ee79ec8a3286517808740a811c0f0ae66c"
dist=metadata.distribution("torch_npu")
stock=Path(dist.locate_file("")).resolve()
record=Path(dist._path)/"RECORD"
native="torch_npu/lib/libtorch_npu.so"
version="torch_npu/version.py"
def file_sha(p):
    with p.open("rb") as inp:
        return hashlib.file_digest(inp,"sha256").hexdigest()
def digest_bytes(digest):
    return "sha256="+base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")

assert dist.version=="2.10.0.post6"
assert (here/"build.exit").read_text().strip()=="0"
assert not Path("/dev/davinci_manager").exists()
wheel_dir=here/"wheel"
wheel_dir.mkdir(exist_ok=True)
name="torch_npu-2.10.0.post6-cp312-cp312-manylinux_2_28_aarch64.whl"
target=wheel_dir/name
payload={}
skipped={}
with WheelFile(str(target),"w") as out, record.open(newline="") as f:
    for key,signature,size in csv.reader(f):
        path=Path(key)
        if "__pycache__" in path.parts or path.suffix==".pyc":
            assert not signature
            skipped["pyc"]=skipped.get("pyc",0)+1
            continue
        if key=="../../../bin/torch_npu_run" or path.parent.name.endswith(".dist-info") and path.name in ("INSTALLER","REQUESTED","RECORD","direct_url.json"):
            skipped[key]=1
            continue
        assert not path.is_absolute() and ".." not in path.parts and path.parts[0] in ("torch_npu","torchnpugen","torch_npu-2.10.0.post6.dist-info"),key
        old=stock/path
        old_digest=file_sha(old)
        assert signature==digest_bytes(old_digest) and old.stat().st_size==int(size),key
        if key==version:
            value=old.read_bytes()
            assert value.count(base.encode())==1
            new=value.replace(base.encode(),head.encode())
            out.writestr(key,new)
            new_digest=hashlib.sha256(new).hexdigest()
        else:
            take=here/"libtorch_npu.so" if key==native else old
            out.write(str(take),key)
            new_digest=file_sha(take) if key==native else old_digest
        payload[key]={"stock_sha256":old_digest,"package_sha256":new_digest}
changed=sorted(k for k,v in payload.items() if len(set(v.values()))!=1)
assert changed==sorted([native,version])
with ZipFile(target) as z:
    records=[f for f in z.namelist() if f.endswith(".dist-info/RECORD")]
    assert len(records)==1
    listed=set()
    for k,check,size in csv.reader(io.StringIO(z.read(records[0]).decode())):
        listed.add(k)
        if k==records[0]:
            assert not check and not size
            continue
        with z.open(k) as src:
            digest=hashlib.file_digest(src,"sha256").hexdigest()
        assert check==digest_bytes(digest) and payload[k]["package_sha256"]==digest and int(size)==z.getinfo(k).file_size,k
    assert set(z.namelist())==listed==set(payload)|set(records)
(here/"package-payload.json").write_text(json.dumps(payload,indent=2)+"\n")
(here/"wheel-receipt.json").write_text(json.dumps(dict(commit=head,base=base,wheel=str(target),wheel_sha256=file_sha(target),
    library_sha256=file_sha(here/"libtorch_npu.so"),stock_elf_sha256=payload[native]["stock_sha256"],entries=len(listed),
    original_record_sha256=file_sha(record),changed=changed,excluded_installer_generated=skipped,
    build_image="sha256:c342a038f6476c794fb364d96479286ebd456021e86dba0d3b715ce4cb983e33"),indent=2)+"\n")
installed=here/"wheel-install"
cmd=[sys.executable,"-B","-m","pip","--isolated","install","--no-index","--disable-pip-version-check","--no-deps","--no-compile","--target",str(installed),str(target)]
print(" ".join(cmd),flush=True)
subprocess.run(cmd,check=True)
overlay=here/"overlay"
overlay.mkdir(exist_ok=True)
if not (overlay/"torch_npu").exists():
    shutil.copytree(installed/"torch_npu",overlay/"torch_npu",copy_function=os.link)
