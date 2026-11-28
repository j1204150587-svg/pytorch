"""Configure three independent endpoint comparisons for the archived HSTU runner."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
experiment = here.parent
study = experiment.parent
head = "e3cda6399cb194c813530222478ea018f5aca5c9"
op_keys = ("NPU_DYNAMIC_REBASE_INDICES", "NPU_DYNAMIC_REBASE_FORWARD")
queue = "TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY"
counter_path = "/artifacts/infra-only-20261128/counter_embedding.so"
counter_sha = "5528151fd5388941af47b417b50b7ced00d2dc661d93425d6008d4d53fb228a7"


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def endpoint(info, enabled, big_cache, counted):
    value = str(int(enabled))
    pkg = info["inner_package"].rstrip("/")
    common = {"PYTHONPATH": pkg, **dict.fromkeys(op_keys, value),
              "NPU_DYNAMIC_REBASE_COUNT": str(int(counted)), "TORCH_NPU_QUEUE_WAKE_AUDIT": str(int(counted))}
    libraries = {pkg + "/torch_npu/lib/libtorch_npu.so": info["sha256"]}
    if big_cache:
        common["ACLNN_CACHE_LIMIT"] = "65536"
    if counted:
        common.update(LD_PRELOAD=counter_path, INFRA_COUNTER_DIR="{output}")
        libraries[counter_path] = counter_sha
    return dict(common=common, switch=queue, A=value, B=value, libraries=libraries)


def main():
    info = json.loads((experiment / "build/ready.json").read_bytes())
    assert info["commit"] == head
    stock = json.loads((study / "recipes/stock.json").read_bytes())
    directory = experiment / "recipes"
    directory.mkdir(exist_ok=True)
    for name in ("fixed-cache", "default-cache", "stock-to-full"):
        large = name != "default-cache"
        for mode in ("performance", "audit"):
            counted = mode == "audit" and name == "fixed-cache"
            a = stock if name == "stock-to-full" else endpoint(info, False, large, counted)
            b = endpoint(info, True, large, counted)
            write(directory / f"{name}-{mode}.json", dict(endpoints={"A": a, "B": b}))

    template = study / "followup70-20261128/stock-to-embedding/endpoint_pair.py"
    source = template.read_text()
    version_check = '        check(sources(stock[0]) == sources(overlay), "overlay contents differ beyond explicitly pinned shared libraries")'
    assert source.count(version_check) == 1
    audit_metadata = f'''        original, actual = sources(stock[0]), sources(overlay)
        check(original.pop("version.py") == {info["stock_version_sha256"]!r}, "stock version digest differs")
        check(actual.pop("version.py") == {info["version_sha256"]!r}, "wheel version digest differs")
        old = (stock[0] / "version.py").read_bytes()
        base = b"38e483ee79ec8a3286517808740a811c0f0ae66c"
        check(old.count(base) == 1 and old.replace(base, {head.encode()!r}) == (overlay / "version.py").read_bytes(),
              "wheel version differs beyond its Git revision")
        check(not list((overlay / "__pycache__").glob("version.*.pyc")), "unexpected cached wheel version")
        check(original == actual, "overlay payload differs beyond the measured library and verified Git revision")'''
    source = source.replace(version_check, audit_metadata)
    anchor = '                namespace = next(line'
    assert source.count(anchor) == 1
    extra = '''                actual_flags = {k: os.fsdecode(env[k.encode()]) if k.encode() in env else None
                                for k in ("ACLNN_CACHE_LIMIT","NPU_DYNAMIC_REBASE_INDICES","NPU_DYNAMIC_REBASE_FORWARD",
                                          "TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY","NPU_DYNAMIC_REBASE_COUNT",
                                          "TORCH_NPU_QUEUE_WAKE_AUDIT")}
                for key,value in actual_flags.items():
                    check(value == expected_env.get(key), f"uncontrolled native option: {key}={value}")
                result["actual_flags"] = actual_flags
'''
    target = here / "endpoint_pair.py"
    target.write_text(source.replace(anchor, extra + anchor))
    target.chmod(0o755)
    write(here / "shim-sha256.json", {p.name + ":" + str(p.parent): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (template, target)})


if __name__ == "__main__":
    main()
