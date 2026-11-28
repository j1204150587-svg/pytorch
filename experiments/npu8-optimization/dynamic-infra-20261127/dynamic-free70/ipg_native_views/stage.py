"""Add one optional host-only strategy to the measured IPG range implementation."""
import hashlib
from pathlib import Path
import shutil
import sys


def stage(ws):
    path=ws/"range_copy.py"
    src=path.read_text(encoding="utf8")
    if hashlib.sha256(src.encode()).hexdigest()!="857e73da2a8d16acf9f29f04bbec78c78231a56a250a872e79d5cc7b9b35f4a4":
        raise RuntimeError("requires original measured IPG range source")
    src=src.replace("class BucketCopy:","from ipg_host_views import HostViews\n\n\nclass BucketCopy:")
    src=src.replace("        self.source_verified=False","        self.source_verified=False\n        self.host_views=HostViews()")
    prep=src.index("        prepared=[]")
    copy=src.index('        if ipg.device.type=="npu":',prep)
    src=src[:prep]+("        plan=self.host_views.prepare(dst,views,z.params_already_reduced)\n"
                   "        if plan is None:\n")+indent(src[prep:copy])+src[copy:]
    src=src.replace("[dst]*(len(prepared)-1)","[dst]*(len(views)-1)")
    first=src.index("        for param_id,g,v in prepared:")
    last=src.index("        z.grads_in_partition_offset=n",first)
    src=src[:first]+("        if plan is None:\n"+indent(src[first:last])+
                     "        else:\n            plan.publish()\n            self.host_views.published+=1\n")+src[last:]
    src=src.replace("    def finish(self,rank):","    def finish(self,rank):\n        self.host_views.finish(rank)")
    compile(src,str(path),"exec")
    path.write_text(src,encoding="utf8")
    shutil.copyfile(Path(__file__).with_name("adapter.py"),ws/"ipg_host_views.py")


def indent(text):
    return "".join("    "+line for line in text.splitlines(keepends=True))


if __name__=="__main__":
    stage(Path(sys.argv[1]))
