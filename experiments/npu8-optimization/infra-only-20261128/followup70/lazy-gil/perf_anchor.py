"""Resolve sampled PTA addresses using each diagnostic worker's mapped ELF and load address."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples",type=Path,help="host-samples folder with .children and worker receipts")
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    result={}
    for leg in ("01-A","02-B"):
        worker=json.loads((args.samples/"pair"/leg/"worker.json").read_bytes())
        name,mapping=next(iter(worker["libraries"].items()))
        begin,end=(int(v,16) for v in mapping.split()[0].split("-"))
        offset=int(mapping.split()[2],16)
        elf=Path("/proc")/str(worker["init_pid"])/"root"/name.lstrip("/")
        with elf.open("rb") as f:
            digest=hashlib.file_digest(f,"sha256").hexdigest()
        assert digest=="4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c"
        parse=re.compile(r"^\s*([\d.]+)%\s+([\d.]+)%\s+(\S+)\s+libtorch_npu.so\s+\[.\] (0x[0-9a-f]+)")
        records=[]
        for line in (args.samples/(leg+".children")).read_text().splitlines():
            match=parse.match(line)
            if match:
                addr=int(match[4],16)
                relative=addr-begin+offset if begin<=addr<end else addr
                if 0<=relative<end-begin+offset:
                    records.append(dict(inclusive_pct=float(match[1]),self_pct=float(match[2]),thread=match[3],
                                        displayed_address=match[4],elf_address=hex(relative)))
        symbols=subprocess.check_output(["addr2line","-C","-f","-e",str(elf)]+[x["elf_address"] for x in records],
                                        text=True).splitlines()
        assert len(symbols)==2*len(records)
        selected=[]
        for row,fn,source in zip(records,symbols[::2],symbols[1::2]):
            if any(needle in fn for needle in ("npu_lazy_init","::empty(","::view(","gil_scoped_acquire::gil_scoped_acquire")):
                selected.append(dict(**row,function=fn,source=source))
        result[leg]={"mapping":mapping,"sha256":digest,"samples":selected}
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__": main()
