"""Export compact raw receipts from completed context-state tests."""
import argparse
import importlib.util
import json
from pathlib import Path
import tarfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("pairs",nargs="*")
    cli=parser.parse_args()
    here=Path(__file__).resolve().parent
    root=here.parent
    spec=importlib.util.spec_from_file_location("export_raw",root/"export.py")
    raw=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw)
    with tarfile.open(cli.output,"w:gz",compresslevel=2,dereference=True) as tar:
        for name in cli.pairs:
            pair=root/"pairs"/name
            r=json.loads((pair/"result.json").read_bytes())
            assert r["exit"]==0
            tar.add(pair,"pairs/"+name)
            source=Path(r["directory"])
            raw.add_raw(tar,source)
            for leaf in source.glob("0?-?/npu_active_context_counter-*.json"):
                tar.add(leaf,"raw/"+source.name+"/"+leaf.relative_to(source).as_posix())
        for path in (root/"recipes/context-state.json",root/"recipes/context-state-audit.json"):
            tar.add(path,"recipes/"+path.name)
        if not cli.pairs or "context-state-audit-ab" in cli.pairs:
            for pattern in ("micro-[01]-[01]","overflow-[01]","timing-[0-3]-[01]"):
                for path in sorted(here.glob(pattern)):
                    assert (path/"train.exit").read_text().strip()=="0"
                    tar.add(path,"probes/"+path.name)
            for path in (here/"mock/mock.json",here/"mock/mock.cpp",here/"build/build-receipt.json",here/"context-state.patch"):
                tar.add(path,path.name)

if __name__=="__main__":
    main()
