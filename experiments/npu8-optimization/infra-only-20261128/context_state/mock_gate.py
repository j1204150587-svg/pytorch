"""Compile the exact modified functions against CPU-only ACL mocks; never reset an NPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from apply import patch


def body(text,signature):
    start=text.index(signature+"\n{")
    end=text.index("\n}",start)+2
    return text[start:end]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("original",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    here=Path(__file__).resolve().parent
    args.out.mkdir(parents=True,exist_ok=True)
    text=patch(args.original.read_text())
    need=["bool hasPrimaryContext(c10::DeviceIndex device_index)","aclError SetDevice(c10::DeviceIndex device)",
          "aclError ResetUsedDevices()","bool isDeviceCtxActive(int32_t device)",
          "aclError MaybeSetDevice(c10::DeviceIndex device)","void LazySetDevice(c10::DeviceIndex device)"]
    functions="\n".join(body(text,k) for k in need)
    prefix=(here/"mock_prelude.inc").read_text()
    generated=prefix+(here/"positive.inc").read_text()+"\n"+functions+"\n"+(here/"mock_cases.inc").read_text()
    source=args.out/"mock.cpp"
    source.write_text(generated)
    exe=args.out/"mock"
    argv=["g++","-std=c++17","-O2","-pthread",str(source),"-o",str(exe)]
    subprocess.run(argv,check=True)
    records=[]
    for toggle in "01":
        env={**os.environ,"NPU_CONTEXT_POSITIVE_MASK":toggle,"NPU_DYNAMIC_REBASE_COUNT":"1"}
        output=subprocess.check_output([str(exe)],env=env,text=True)
        records.append(dict(switch=toggle,checks=output.splitlines()))
    (args.out/"mock.json").write_text(json.dumps(dict(compile=argv,real_acl=False,records=records),indent=2)+"\n")


if __name__=="__main__":
    main()
