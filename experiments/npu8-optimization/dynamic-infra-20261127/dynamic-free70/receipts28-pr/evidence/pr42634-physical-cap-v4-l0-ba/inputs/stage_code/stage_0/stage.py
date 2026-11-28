"""Add the opt-in eager side-table primitive to a staged training workspace."""
import argparse
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"expected one match for {old!r}")
    return source.replace(old, new)


def stage(workspace):
    path = workspace / "model.py"
    source = path.read_text(encoding="utf8")
    source = replace(source, "import math\n", "import math\nimport os\n")
    anchor = "        self.output.weight = self.item_embedding.weight\n"
    source = replace(source, anchor, anchor + """        self._side_calls = self._side_hits = 0
        if os.environ.get("NPU_DYNAMIC_GROUPED_SIDE") == "1":
            offsets = torch.arange(config.num_tables, dtype=torch.int64) * config.table_rows
            self.register_buffer("_side_offsets", offsets.view(-1, 1, 1, 1), persistent=False)
""")
    anchor = """        side = None
        for table_index, table in enumerate(self.side_embeddings):
            table_features = table(feature_ids[:, :, table_index, :]).mean(dim=-2)
            side = table_features if side is None else side + table_features
"""
    source = replace(source, anchor, """        side = None
        if hasattr(self, "_side_offsets"):
            from side_tables_op import model_side_sum
            side = model_side_sum(self, feature_ids)
        if side is None:
            for table_index, table in enumerate(self.side_embeddings):
                table_features = table(feature_ids[:, :, table_index, :]).mean(dim=-2)
                side = table_features if side is None else side + table_features
""")
    compile(source, str(path), "exec")
    train = workspace / "train_deepspeed.py"
    if train.exists():
        ts = train.read_text(encoding="utf8")
        pos = "        end = time.perf_counter()\n"
        ts = replace(ts, pos, pos + """        (rank_dir / "grouped_side_counts.json").write_text(json.dumps({
            "enabled": hasattr(model, "_side_offsets"), "calls": model._side_calls,
            "hits": model._side_hits, "fallbacks": model._side_calls - model._side_hits,
            "table_parameters": len(model.side_embeddings)
        }, indent=2))
""")
        compile(ts, str(train), "exec")
        train.write_text(ts, encoding="utf8")
    path.write_text(source, encoding="utf8")
    for name in ("npu_ordered_fold.py", "side_tables_op.py"):
        shutil.copyfile(HERE / name, workspace / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="new staged training workspace")
    stage(parser.parse_args().workspace)
