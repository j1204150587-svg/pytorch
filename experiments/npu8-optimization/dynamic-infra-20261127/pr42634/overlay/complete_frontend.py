"""Install exact upstream Python dependencies missing from the older wheel into the overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile


def main(out):
    out = out.resolve(strict=True)
    manifest = json.loads((out/"receipt.json").read_text())
    target = (out/"package").resolve(strict=True)
    if target.parent != out or manifest["head"] != "2701f7b3e8d89625752edbde10e42c2b2172272c":
        raise ValueError("expected the isolated PR 42634 package")
    blobs = {"torch_npu/utils/_dynamo.py":"f6c503c9cf91a55ed4a521d42abce2c6db23c266",
             "torch_npu/dynamo/_deterministic_guard.py":"7ef431d09ec151108bf5b2efec167d7fb9820817"}
    archive = Path(__file__).with_name("pr-frontend-deps.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != "081a081b124b2b68b553030136e70787fb7a5cf73195ed598bec8bf4e3bad3a8":
        raise ValueError("unexpected frontend dependency archive")
    with tarfile.open(archive) as tar:
        for name,blob in blobs.items():
            data = tar.extractfile(name).read()
            if hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest() != blob:
                raise ValueError("unexpected source blob " + name)
            path = target/name
            if not path.parent.resolve().is_relative_to(target):
                raise ValueError("package dependency path escapes overlay")
            # Directory entry replacement keeps baseline hard links untouched.
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
                f.write(data)
            Path(f.name).replace(path)
    (out/"frontend-dependencies.json").write_text(json.dumps(blobs,indent=2)+"\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output",type=Path)
    main(cli.parse_args().output)
