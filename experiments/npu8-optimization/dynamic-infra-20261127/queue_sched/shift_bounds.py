"""Exact extrema of interval overlap when every scheduler timestamp shares one clock offset."""
from bisect import bisect_right
from collections import defaultdict


def merge(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = result[-1][0], max(result[-1][1], b)
        else:
            result.append((a, b))
    return result


def extrema(observed, states, low, high):
    # Overlap of [a,b] and [c+x,d+x] equals (b-c-x)+ - (a-c-x)+
    # - (b-d-x)+ + (a-d-x)+, where y+ = max(y,0). Its slope changes only
    # at these four differences. Scan all changes in the feasible offset range.
    starts = [a for a, _ in states]
    ends = [b for _, b in states]
    changes = defaultdict(int)
    value = slope = 0
    for a, b in observed:
        i = bisect_right(ends, a-high)
        while i < len(states) and starts[i] < b-low:
            c, d = states[i]
            for knot, sign in ((b-c, 1), (a-c, -1), (b-d, -1), (a-d, 1)):
                if knot > low:
                    value += sign * (knot-low)
                    slope -= sign
                    if knot < high:
                        changes[knot] += sign
            i += 1
    floor = ceiling = value
    cursor = low
    for edge in sorted(changes):
        value += slope * (edge-cursor)
        floor, ceiling = min(floor, value), max(ceiling, value)
        slope += changes[edge]
        cursor = edge
    value += slope * (high-cursor)
    return min(floor, value), max(ceiling, value)


def bound_rows(raw_states, observed, low, high, quantum):
    output = []
    if not raw_states:
        raise ValueError("no scheduler observations")
    first = raw_states[0][0] + high + quantum
    last = raw_states[-1][1] + low - quantum
    if any(a < first or b > last for windows in observed.values() for a,b in windows):
        raise ValueError("the scheduler capture does not bracket the device-gap windows")
    for label in ("scheduled", "off_blocked", "runnable_after_wakeup",
                  "runnable_switch_out_R", "runnable_after_preempt"):
        intervals = merge((a,b) for a,b,state in raw_states if state == label)
        # ftrace has finite precision at each boundary independently. Erode or
        # expand, then merge again before intersecting to avoid double counting.
        inner = merge((a+quantum, b-quantum) for a,b in intervals)
        uncertain_off = [(a,b) for a,b,state in raw_states if state.startswith("off_unknown") and label != "scheduled"]
        outer = merge((a-quantum, b+quantum) for a,b in intervals + uncertain_off)
        for (step,phase), windows in sorted(observed.items()):
            windows = merge(windows)
            lower = extrema(windows, inner, low, high)[0]
            upper = extrema(windows, outer, low, high)[1]
            total = sum(b-a for a,b in windows)
            if not 0 <= lower <= upper <= total:
                raise ValueError("invalid scheduling-state bounds")
            output.append(dict(step=step, phase=phase, state=label, phase_ms=total/1e6,
                               lower_ms=lower/1e6, upper_ms=upper/1e6))
    return output
