"""Offline regression using previously collected AB/BA training legs as runner fixtures."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import ordered_resume as o


def test(fixture, upstream, registered_name):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        family=root/"runs"/registered_name
        family.mkdir(parents=True)
        shutil.copyfile(fixture/"registries"/registered_name/"recipe.json",family/"recipe.json")
        shutil.copytree(upstream,family/"runner")
        (family/"gate").mkdir()
        history=next((fixture/"gates"/registered_name).rglob("comparison.json"))
        shutil.copyfile(history,family/"gate/comparison.json")
        o.save(family/"runner-sha256.json",{f.name:o.file_hash(f) for f in (family/"runner").glob("*.py")})
        args=SimpleNamespace(sample_every=2,max_sample_gap=8,poll=0.001,container_id="demo")
        path=root/"campaign"
        path.mkdir()
        calls=[]
        current={}
        sys.path.insert(0,str(upstream))
        import run_npu8
        synthetic=lambda t,procs=(): {"utc":datetime.fromtimestamp(t,timezone.utc).isoformat(),"valid":True,
                                       "processes":list(procs)}

        def original(argv,stdout,**kwargs):
            assert "DYNAMIC_INFRA_RECIPE" in kwargs["env"] and kwargs["pass_fds"]==(99,)
            with patch.object(sys,"argv",argv[2:]):
                cli=run_npu8.parse_args()
            v={"baseline":"A","screen":"B"}[cli.stage]
            flags,_=run_npu8.training_options(cli,"/artifacts/test")
            assert (flags["steps"],flags["steady_warmup_steps"],flags["steady_measure_steps"],
                    flags["profile_active"])==((60,10,50,5) if cli.mode=="no-profiler" else (8,0,0,3))
            mode,order=current["mode"],current["order"]
            sample=fixture/"evidence"/f"{registered_name}-{mode}-{order}"
            source=next(sample.glob("[0-9][0-9]-"+v))
            session=root/cli.approval_ref
            session.mkdir()
            leg=session/("01-"+v)
            shutil.copytree(source,leg)
            for key in ("container.json","workspace.sha256"):
                shutil.copyfile(sample/key,session/key)
            if (sample/"workspace-01.sha256").exists():
                shutil.copyfile(sample/"workspace-01.sha256",session/"workspace-01.sha256")
            o.save(session/"plan.json",{"order":[v],"mode":mode,"approval_ref":cli.approval_ref})
            o.save(session/"status.json",{"exit":0,"phase":"completed"})
            command=o.read(leg/"command.json")
            command["workspace"]="/artifacts/test/"+session.name+"/"+Path(command["workspace"]).name
            o.save(leg/"command.json",command)
            a=o.epoch(o.read(leg/"before.card.json")["utc"])
            b=o.epoch(o.read(leg/"after.card.json")["utc"])
            rows=[synthetic(t+0.25) for t in range(int(a),int(b)+1,2)]
            if len(calls)==1:  # The first B finishes but the independent sample detected a foreign user.
                rows[len(rows)//2]["processes"]=[{"pid":321234,"owner":"external"}]
            current["rows"]=rows
            calls.append((mode,order,v))
            stdout.write((str(session)+"\n").encode())
            return SimpleNamespace(returncode=0)

        def observe(stop,container,log,interval):
            assert stop.wait(30)
            log.write_text("".join(json.dumps(v)+"\n" for v in current["rows"]))

        with patch.object(o,"FREE70",root),patch.object(o,"observe",observe), \
             patch.object(o,"safe_sample",lambda _:synthetic(0)),patch.object(o.subprocess,"run",original):
            plan=o.make_plan([registered_name],["no-profiler","l0"])
            source=plan["families"][registered_name]
            expected=[]
            for j,job in enumerate(plan["jobs"]):
                current.update(job)
                parent=path/f"pair-{j}"
                parent.mkdir()
                def execute():
                    return [o.run_slot(parent/f"{i}-{v}",job,v,source,99,args)
                            for i,v in enumerate(job["order"].upper())]
                legs=execute()
                expected.extend((job["mode"],job["order"],v) for v in job["order"].upper())
                if j==0:
                    expected.insert(1,(job["mode"],job["order"],"B"))
                total=len(calls)
                assert [v["token"] for v in execute()]==[v["token"] for v in legs] and total==len(calls)
                target=o.install_pair(path,job,legs,source)
                assert o.read(target/"status.json")["exit"]==0
                assert [e["variant"] for e in o.read(target/"pair-provenance.json")["legs"]]==list(job["order"].upper())
            assert calls==expected
            first=path/"pair-0/1-B/try-00001/decision.json"
            assert o.read(first)["verdict"]=="external"
            quiet=[synthetic(2)]
            bounds=({"utc":synthetic(0)["utc"]},{"utc":synthetic(7)["utc"]})
            assert o.audit_samples(*bounds,[],8)[0]=="observation_failed"
            assert o.audit_samples(*bounds,quiet,4)[0]=="observation_failed"
            quiet[0]["valid"]=False
            assert o.audit_samples(*bounds,quiet,8)[0]=="observation_failed"
            print("verified original runner options, 4 ordered pairs, 8 accepted full legs, 1 rejected B, restart deduplication and sample coverage")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("fixture",type=Path)
    cli.add_argument("upstream",type=Path)
    cli.add_argument("--family",default="grouped-side-v1")
    a=cli.parse_args()
    test(a.fixture.resolve(),a.upstream.resolve(),a.family)
