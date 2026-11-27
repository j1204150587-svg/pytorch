from pathlib import Path
import subprocess
import sys

if __name__=="__main__":
    root=Path(__file__).resolve().parent
    for stage in sorted(root.glob("stage_*"),key=lambda p:int(p.name.split("_")[1])):
        subprocess.run([sys.executable,"-B",str(stage/"stage.py"),sys.argv[1]],check=True)
