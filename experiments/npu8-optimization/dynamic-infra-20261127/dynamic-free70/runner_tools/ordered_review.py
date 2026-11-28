"""Independently check registration, stage identity, occupancy samples and time gaps from an exported campaign."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

from ordered_resume import audit_samples, epoch


def review(archive):
    with tarfile.open(archive) as packed:
        members={m.name:m for m in packed if m.isfile()}
        def raw(name):
            return packed.extractfile(members[name]).read()
        def read(name):
            return json.loads(raw(name))
        manifest=read("file-sha256.json")
        assert set(manifest)==set(members)-{"file-sha256.json"}
        for key,row in manifest.items():
            assert members[key].size==row["bytes"]
            digest=hashlib.sha256()
            with packed.extractfile(members[key]) as file:
                for block in iter(lambda:file.read(2**20),b""):
                    digest.update(block)
            assert digest.hexdigest()==row["sha256"],key
        plan=read("plan.json")
        runs=read("runs.json")
        specs=plan["jobs"]
        slots={}
        for name in members:
            if name.startswith("attempts/") and name.endswith("/decision.json"):
                folder=name.rsplit("/",1)[0]
                req=read(folder+"/request.json")
                decision=read(name)
                token=req["token"]
                assert token not in slots and decision["token"]==token
                j=specs.index({k:req[k] for k in ("family","mode","order")})
                pair,slot,attempt=folder.split("/")[1:]
                assert pair==f"{j+1:02d}-{req['family']}-{req['mode']}-{req['order']}"
                assert slot[3:]==req["variant"]==req["order"][int(slot[:2])-1].upper()
                slots[token]=(folder,req,decision)
        summaries=[]
        for run in runs:
            assert run["exit"]==0
            family,mode,order=[run[k] for k in ("family","mode","order")]
            key=f"evidence/{family}-{mode}-{order}/"
            prov=read(key+"pair-provenance.json")
            source=plan["families"][family]
            assert prov["source"]==source and prov["plan"]=={k:run[k] for k in ("family","mode","order")}
            assert manifest[f"gates/{family}/comparison.json"]["sha256"]==source["gate_sha256"]
            assert manifest[f"registries/{family}/recipe.json"]["sha256"]==source["recipe_sha256"]
            register=read(f"registries/{family}/recipe.json")
            rows=[]
            for index,leg in enumerate(prov["legs"],1):
                variant=order[index-1].upper()
                slot=key+f"{index:02d}-{variant}/"
                prefix,req,stored=slots[leg["token"]]
                assert req["variant"]==variant and stored["verdict"]=="accepted"
                assert all(leg[k]==v for k,v in stored.items())
                siblings=[x for x in slots.values() if x[0].rsplit("/",1)[0]==prefix.rsplit("/",1)[0]]
                assert len(siblings)==leg["attempt_count"]
                assert sorted(x[2]["verdict"] for x in siblings)==["accepted"]+["external"]*(len(siblings)-1)
                assert read(slot+"recipe.json")==register and read(slot+"command.json")==leg["command"]
                assert read(slot+"rank000/runtime.json")==leg["runtime"]
                assert read(slot+"rank000/run_arguments.json")==leg["actual_args"]
                assert all(int(raw(slot+name+".exit"))==0 for name in ("preflight","train","postflight","analysis","runner"))
                before,after=[read(slot+n+".card.json") for n in ("before","after")]
                samples=[json.loads(line) for line in raw(prefix+"/chip.jsonl").splitlines()]
                verdict,counts=audit_samples(before,after,samples,plan["max_sample_gap"])
                assert verdict=="accepted" and counts==stored["audit"]
                rows.append({"variant":variant,"attempts":len(siblings),"samples":counts["sample_count"],
                             "max_sample_gap_s":counts["max_gap_seconds"],"before":before["utc"],"after":after["utc"],
                             "workspace_sha256":leg["workspace_sha256"]})
            gap=epoch(rows[1]["before"])-epoch(rows[0]["after"])
            assert gap>=0 and gap==prov["gap_seconds_at_seconds_resolution"]
            summaries.append({"family":family,"mode":mode,"order":order,"gap_seconds":gap,"legs":rows})
        return {"archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"file_count":len(manifest),
                "decided_attempts":dict(Counter(x[2]["verdict"] for x in slots.values())),
                "registered_families":list(plan["families"]),"pairs":summaries}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    args.output.write_text(json.dumps(review(args.archive),indent=2)+"\n",encoding="utf8")
