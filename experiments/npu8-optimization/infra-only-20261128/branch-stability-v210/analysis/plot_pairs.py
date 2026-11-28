"""Plot every accepted synchronized and Level0 pair from measure.py (requires Matplotlib)."""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

HEAD = "e3cda6399cb194c813530222478ea018f5aca5c9"
GROUPS = {
    "fixed-cache": ("同包大缓存", "三个开关全关 → 全开，缓存均为 65,536"),
    "stock-to-full": ("原镜像到完整配置", "原镜像默认缓存 → 新包全开、缓存 65,536"),
    "default-cache": ("同包默认缓存", "三个开关全关 → 全开，均沿用默认缓存"),
}
MODES = {"no-profiler": 6, "l0": 4}
COLORS = {"ab": "#00727D", "ba": "#B25024"}
MARKERS = {"ab": "o", "ba": "D"}


@dataclass(frozen=True)
class Point:
    group: str
    mode: str
    repeat: int
    order: str
    gain: float


def read(path):
    return json.loads(path.read_bytes())


def measure(row, mode):
    if mode == "no-profiler":
        if (row["samples"], row["tokens"]) != (136, 2994):
            raise ValueError("unexpected measured workload")
        return row["seconds"] * 1000 / 50
    steps = row["steps"]
    if [(s["step"], s["shape"]) for s in steps] != [(4, [1, 19]), (5, [2, 14]), (6, [2, 22])]:
        raise ValueError("unexpected Level0 steps or actual shapes")
    return sum(s["Free"] for s in steps) / sum(s["Stage"] for s in steps)


def points_from(root):
    schedule = read(root / "data/expected-schedule.json")
    expected = {s["id"]: s for s in schedule if s["mode"] in MODES}
    points = []
    seen = set()
    builds = set()
    for group in GROUPS:
        report = read(root / "reports" / group / "metrics.json")
        if report["comparison"] != group or report["branch_commit"] != HEAD:
            raise ValueError("comparison or branch fingerprint differs")
        builds.add(report["branch_library_sha256"])
        for row in report["pairs"]:
            mode, key = row["mode"], row["id"]
            if mode not in MODES:
                continue
            spec = expected[key]
            if (key in seen or (spec["comparison"], spec["mode"], spec["order"]) != (group, mode, row["order"])
                    or row["proof"]["status"] != "host_receipt" or set(row["proof"]["chip"]) != {"A", "B"}):
                raise ValueError(f"duplicate, mismatched or unverified pair {key}")
            a, b = (measure(row[arm], mode) for arm in "AB")
            gain = 100 * (1-b/a)
            recorded = row["gain_pct" if mode == "no-profiler" else "free_ratio_gain_pct"]
            if not math.isclose(gain, recorded, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"recomputed gain differs for {key}")
            points.append(Point(group, mode, spec["repeat"], spec["order"], gain))
            seen.add(key)
    if seen != set(expected):
        raise ValueError(f"plot requires all 30 accepted timing pairs; available {len(seen)}/30")
    if len(builds) != 1 or not next(iter(builds)):
        raise ValueError("branch library differs across groups")
    for group in GROUPS:
        for mode, n in MODES.items():
            series = [p for p in points if (p.group, p.mode) == (group, mode)]
            if (sorted(p.repeat for p in series) != list(range(1, n+1))
                    or Counter(p.order for p in series) != {"ab": n//2, "ba": n//2}):
                raise ValueError(f"incomplete or inconsistent repeats: {group} {mode}")
    return points


def plot(points, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    for family in ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "SimHei"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("install a Simplified Chinese font such as Noto Sans CJK SC")
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "axes.axisbelow": True,
                         "axes.labelcolor": "#334353", "text.color": "#203142", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.2), sharey="row")
    fig.subplots_adjust(left=.083, right=.966, bottom=.192, top=.765, hspace=.40, wspace=.18)
    fig.text(.055, .941, "动态 HSTU 训练的逐轮配对收益", size=24, weight="bold",
             ha="left", va="center")
    fig.text(.056, .883, "逻辑 NPU 8  ·  60 个合成 shape  ·  点上的数值为相对降幅（%），正值表示 B 改善",
             size=11, color="#485B6B", va="center")
    handles = [Line2D([], [], linestyle="none", marker=MARKERS[o], color=COLORS[o], markersize=7.5)
               for o in ("ab", "ba")]
    fig.legend(handles, ["AB（先 A 后 B）", "BA（先 B 后 A）"], loc="center right", ncol=2,
               bbox_to_anchor=(.977,.94), frameon=False, fontsize=10, handletextpad=.28, columnspacing=1.1)
    for col, group in enumerate(GROUPS):
        ax = axes[0, col]
        ax.text(.5, 1.25, GROUPS[group][0], transform=ax.transAxes, ha="center", va="center",
                size=14.5, weight="bold")
        ax.text(.5, 1.105, GROUPS[group][1], transform=ax.transAxes, ha="center", va="center",
                size=9, color="#53697A")
    for i, (mode, n) in enumerate(MODES.items()):
        values = [p.gain for p in points if p.mode == mode]
        low, high = min(0, min(values)), max(0, max(values))
        span = max(high-low, .4)
        for col, group in enumerate(GROUPS):
            ax = axes[i, col]
            ax.set_xlim(.52, n+.48)
            ax.set_ylim(low-.15*span, high+.24*span)
            ax.set_xticks(range(1, n+1))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
            ax.grid(axis="y", color="#DCE3E7", linestyle=":", linewidth=.8)
            ax.axhline(0, color="#526373", lw=1.1)
            ax.tick_params(which="both", length=0, pad=5, labelsize=9, colors="#425668")
            for side in ("top", "left", "right"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#BCC7CF")
            if col == 0:
                ax.set_ylabel(("同步步时降幅（%）" if i==0 else "Free 占比相对降幅（%）"), fontsize=10.5, labelpad=8)
            if i == 1:
                ax.set_xlabel("配对轮次", fontsize=9.5, labelpad=6)
            for p in points:
                if (p.group,p.mode)!=(group,mode):
                    continue
                color=COLORS[p.order]
                ax.plot(p.repeat, p.gain, marker=MARKERS[p.order], markersize=7.5, linestyle="none",
                        markerfacecolor=color, markeredgecolor="white", markeredgewidth=.8)
                ax.annotate(f"{p.gain:+.2f}", (p.repeat,p.gain), xytext=(0,8), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.2, color=color)
    fig.text(.056, .093, "上排取无 profiler 的 50 步同步窗，每组 6 个配对。下排取 Level0 第 4–6 步，每组 4 个独立配对。",
             fontsize=10.3, color="#485B6B")
    fig.text(.056, .050, "每点按 100 × (1 − B/A) 计算。下排先在各腿求 ΣFree/ΣStage，再计算该占比的相对降幅。",
             fontsize=10.1, color="#485B6B")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name+".tmp")
    fig.savefig(temp, format="png", dpi=180, metadata={"Title":"动态 HSTU 训练的逐轮配对收益","Source":HEAD})
    plt.close(fig)
    temp.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    output = root/"reports/paired-runs.png"
    plot(points_from(root), output)
    print(output)


if __name__ == "__main__":
    main()

