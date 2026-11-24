"""Generate the lookahead patch from an archived 26 training workspace."""

from difflib import unified_diff
from pathlib import Path
import sys


def change(code, old, new):
    if code.count(old) != 1:
        raise ValueError(f"expected one source anchor: {old}")
    return code.replace(old, new)


def graph(code):
    code = change(code, '    class Replay(torch.autograd.Function):\n', '''    hook_lookahead = None
    if os.environ.get("RECSYS_HSTU_FAST_GUARDS") == "1":
        if direct is None:
            raise RuntimeError("replay lookahead requires 26 direct gradients")
        from graph_guards import HookLookahead
        from torch.optim import optimizer as torch_optimizer

        # This fixed-input study owns the model and all changes to its hook
        # registries. Reject arbitrary optimizer callbacks in the interval.
        local_optimizer = engine.optimizer.optimizer
        def require_fixed_optimizer():
            if (engine.optimizer.optimizer is not local_optimizer
                    or type(local_optimizer) is not torch.optim.AdamW
                    or local_optimizer._optimizer_step_pre_hooks
                    or local_optimizer._optimizer_step_post_hooks
                    or torch_optimizer._global_optimizer_pre_hooks
                    or torch_optimizer._global_optimizer_post_hooks):
                raise RuntimeError("replay lookahead requires the fixed AdamW callback path")

        hook_lookahead = HookLookahead(lambda: check_backward_hooks(original, parameters), require_fixed_optimizer)
        original._recsys_prepare_replay = lambda: hook_lookahead.prepare(direct.generation + 1)

    class Replay(torch.autograd.Function):
''')
    code = change(code, '            check_backward_hooks(original, parameters)\n            return Replay.apply',
                  '''            if hook_lookahead is None:
                check_backward_hooks(original, parameters)
            else:
                hook_lookahead.consume(direct.generation + 1)
            return Replay.apply''')
    return change(code, '        (Path(rank_dir) / "hstu_graph.json").write_text',
                  '''        record["fast_guards"] = hook_lookahead is not None
        if hook_lookahead is not None:
            record["hook_lookahead"] = hook_lookahead.finish(expected)
        (Path(rank_dir) / "hstu_graph.json").write_text''')


def training(code):
    code = change(code, '    drained = False\n', '''    drained = False
    prepare_replay = getattr(engine.module, "_recsys_prepare_replay", None)
''')
    return change(code, '                            overlap_input.submit(next_step, input_audit)\n',
                  '''                            overlap_input.submit(next_step, input_audit)
                            if prepare_replay is not None:
                                prepare_replay()
''')


def main():
    workspace, output = map(Path, sys.argv[1:])
    with output.open("w", encoding="utf-8", newline="\n") as patch:
        for name, edit in (("graphed_hstu.py", graph), ("train_deepspeed.py", training)):
            source = (workspace / name).read_text(encoding="utf-8")
            patch.writelines(unified_diff(source.splitlines(keepends=True), edit(source).splitlines(keepends=True),
                                          fromfile=f"a/{name}", tofile=f"b/{name}"))
        module = Path(__file__).with_name("graph_guards.py").read_text(encoding="utf-8")
        patch.writelines(unified_diff([], module.splitlines(keepends=True),
                                      fromfile="/dev/null", tofile="b/graph_guards.py"))


if __name__ == "__main__":
    main()
