"""Stage the measured 08 training inputs and a semantic gate in an isolated run directory."""

from pathlib import Path
import shutil
import sys


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent
STUDY = ARTIFACTS / "dispatch-study-20261119-npu8"
STEADY = STUDY / "20261123T123103Z-ba-no-profiler-w_xzf07x"
PROFILE = STUDY / "20261123T123759Z-ab-l0-dysou58k"
SEMANTICS = ARTIFACTS / "pta-semantics-uhpdqhpa" / "tools"


def modify(path, old, new):
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"unexpected source in {path}: {old}")
    path.write_text(source.replace(old, new), encoding="utf-8")


def stage_verifier():
    verifier = HERE / "validate_npu.py"
    shutil.copy2(SEMANTICS / "validate_npu.py", verifier)
    modify(verifier, 'CASES = ("finite_below_clip", "nan", "posinf", "neginf", "finite_recovery_above_clip")',
           '''REAL_CASES = ("real_gradient_1", "real_gradient_2", "real_gradient_3")
CASES = ("finite_below_clip", "nan", "posinf", "neginf", "finite_recovery_above_clip") + REAL_CASES''')
    source = verifier.read_text(encoding="utf-8")
    start = source.index("            with torch.no_grad():\n                for grad in grads:")
    end = source.index("\n            skipped_before = engine.skipped_steps", start)
    block = source[start:end]
    injection = "\n".join("    " + line if line else "" for line in block.split("\n"))
    modify(verifier, block, "            if case not in REAL_CASES:\n" + injection)
    modify(verifier, "elif case == CASES[-1]:", 'elif case == "finite_recovery_above_clip":')
    modify(verifier, '                require(all((event["reference_clip_factor"] < 1) == (case == CASES[-1])\n'
                     '                            for event in clip_events), f"{case}: wrong side of clip threshold")',
           '''                if case not in REAL_CASES:
                    require(all((event["reference_clip_factor"] < 1) == (case == "finite_recovery_above_clip")
                                for event in clip_events), f"{case}: wrong side of clip threshold")''')
    modify(verifier, 'len(norm_events) == 2', 'len(norm_events) == len(CASES) - 3')
    modify(verifier, 'norm_stats.require_batched_step(2)', 'norm_stats.require_batched_step(len(CASES) - 3)')
    modify(verifier, '            invalid = case in ("nan", "posinf", "neginf")', """            cursor = grads[0].storage_offset()
            storage = grads[0].untyped_storage().data_ptr()
            adjacent = True
            for grad in grads:
                adjacent &= grad.untyped_storage().data_ptr() == storage and grad.storage_offset() == cursor
                cursor += grad.numel()
            report.setdefault("adjacent_gradient_partitions", []).append(adjacent)
            invalid = case in ("nan", "posinf", "neginf")""")
    modify(verifier, "            engine.step()\n            torch.npu.synchronize()", """            if os.environ.get("NPU_EXPERIMENT_FLAT_ND") == "1" and not invalid:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    engine.step()
                    torch.npu.synchronize()
                hits = sum(event.name == "NPUDenseFlattenStorageCopy" for event in prof.events())
                report.setdefault("native_flatten_hits", []).append(hits)
                require(hits == 1, "no unique native flat copy on this ZeRO step")
            else:
                engine.step()
                torch.npu.synchronize()""")
    compile(verifier.read_text(encoding="utf-8"), str(verifier), "exec")


def main():
    if sys.argv[1:] == ["--verifier-only"]:
        stage_verifier()
        return
    for name in ("run_npu8.py", "evidence.py"):
        shutil.copy2(STEADY / "inputs" / name, HERE / name)
    shutil.copy2(STEADY / "01-B/command.json", HERE / "steady-command.json")
    shutil.copy2(PROFILE / "02-B/command.json", HERE / "level0-command.json")
    work = HERE / "work"
    shutil.copytree(STEADY / "workspace", work)
    source = work / "pta_graph_function.py"
    modify(source, "    def run(value):", """    import os

    queued = os.environ.get("NPU_EXPERIMENT_QUEUE_COPY_REPLAY") == "1"

    def run(value):""")
    modify(source, "        static.copy_(value)\n        captured.replay()", """        if queued:
            if not captured._try_replay_with_copy(static, value):
                raise RuntimeError("queued 08 staging/replay rejected the runtime layout or stream")
        else:
            static.copy_(value)
            captured.replay()""")
    stage_verifier()


if __name__ == "__main__":
    main()
