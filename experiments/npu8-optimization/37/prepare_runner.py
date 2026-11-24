"""Prepare a private copy of the 26 runner for the replay lookahead A/B."""

from pathlib import Path
import sys

from make_patch import change


def main():
    source, target = map(Path, sys.argv[1:])
    code = source.read_text(encoding="utf-8")
    code = change(code, '== {"graphed_hstu.py"},\n                "graph guards patch must change only the replay and guard module")',
                  '== {"graphed_hstu.py", "train_deepspeed.py"},\n'
                  '                "lookahead patch must change only replay, training entry and guard module")')
    code = change(code, 'if args.extension_patch and args.stage not in {"ab", "ba", "screen"}:',
                  'if args.extension_patch and args.stage not in {"ab", "ba", "screen"} and not (\n'
                  '                args.graph_guards_patch and args.stage == "aa"):')
    code = change(code, '        if args.stage in {"ab", "ba", "screen"}:\n',
                  '        if args.stage in {"ab", "ba", "screen"} or args.graph_guards_patch:\n')
    code = change(code,
                  '            require(graph["fast_guards"] is (variant == "B"), "fast graph guards activation disagrees with experiment")',
                  '''            require(graph["fast_guards"] is (variant == "B"), "lookahead activation disagrees with experiment")
            lookahead = graph.get("hook_lookahead")
            require((lookahead is not None) is (variant == "B"), "lookahead receipt missing or unexpected")
            if lookahead is not None:
                hits = flags["steps"] - 1 - int(bool(flags["steady_measure_steps"]))
                require(lookahead["submitted"] == lookahead["consumed"] == hits
                        and lookahead["synchronous"] + hits == flags["steps"]
                        and 0 <= lookahead["ready"] <= hits,
                        "lookahead generation counts differ from the input overlap")
                result["hook_lookahead"] = lookahead''')
    compile(code, str(target), "exec")
    target.write_text(code, encoding="utf-8")


if __name__ == "__main__":
    main()
