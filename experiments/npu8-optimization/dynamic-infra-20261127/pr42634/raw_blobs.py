"""Rebuild the two source archives from Git blobs, independent of Windows checkout EOL."""
import gzip
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tarfile


HEAD = "2701f7b3e8d89625752edbde10e42c2b2172272c"


def main(directory):
    for filename, script in (("pr-source.tar.gz","build.py"), ("pr-frontend-deps.tar.gz","complete_frontend.py")):
        path = directory / filename
        old = hashlib.sha256(path.read_bytes()).hexdigest()
        with tarfile.open(path) as tar:
            members = [x.name for x in tar.getmembers() if x.isfile()]
        queries = "".join(HEAD+":"+name+"\n" for name in members).encode()
        raw = subprocess.check_output(["git","cat-file","--batch"],input=queries)
        stream = io.BytesIO(raw)
        packed = io.BytesIO()
        with gzip.GzipFile(fileobj=packed,mode="wb",mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode="w") as archive:
                for name in members:
                    oid, typ, n = stream.readline().split()
                    if typ != b"blob":
                        raise ValueError(name)
                    data = stream.read(int(n))
                    if stream.read(1) != b"\n" or hashlib.sha1(b"blob "+n+b"\0"+data).hexdigest().encode() != oid:
                        raise ValueError(name)
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
        target = packed.getvalue()
        new = hashlib.sha256(target).hexdigest()
        source = (directory / script).read_text()
        if old not in source:
            raise ValueError("archive hash absent from "+script)
        path.write_bytes(target)
        (directory / script).write_text(source.replace(old,new), encoding="utf8",newline="\n")
        print(filename,len(members),new)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
