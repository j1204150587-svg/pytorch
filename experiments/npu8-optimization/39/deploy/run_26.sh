#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage:\n  bash run_26.sh correctness /new/absolute/out\n  bash run_26.sh {steady|level0} /new/absolute/out {aa|ab|ba} /absolute/correctness/comparison.json\n' >&2
    exit 2
}
[[ $# -ge 2 && $2 == /* && ! -e $2 ]] || usage
mode=$1
out=$2
case $mode in
    correctness) [[ $# == 2 ]] || usage; order=ab ;;
    steady|level0) [[ $# == 4 && $3 =~ ^(aa|ab|ba)$ && $4 == /* ]] || usage; order=$3 ;;
    *) usage ;;
esac
[[ ${RECSYS_NPU_IDS:-} =~ ^[0-9]+$ ]] || {
    printf 'Select one reserved NPU in the container via RECSYS_NPU_IDS.\n' >&2; exit 2;
}
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
bundle=$PWD
sha256sum --quiet --check bundle.sha256
read -r fingerprint _ < <(sha256sum bundle.sha256)
if [[ $mode != correctness ]]; then
    python3 -B compare26.py check-gate "$4" "$fingerprint"
fi
compare_arguments=(--order "$order")
if [[ $mode != correctness ]]; then
    compare_arguments+=(--gate "$4")
fi

export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export RECSYS_ZERO2_BATCH_OVERFLOW=1 RECSYS_HSTU_COMBINED=1 RECSYS_PTA_WRAPPER=1
export RECSYS_PTA_REPLAY=1 RECSYS_PTA_SCOPE=both RECSYS_ZERO2_LOCAL_STEP=both
export RECSYS_HSTU_GRAPH_SIDEPACK=1 RECSYS_HSTU_DIRECT_GRADIENT=1 RECSYS_OVERLAP_INPUT=1
export RECSYS_OVERLAP_INPUT_AUDIT=0 RECSYS_26_INTEGRATION_MODE=$mode
common=(--model-config configs/model_many_tables.json
        --deepspeed-config configs/deepspeed_zero2_bf16_overflow_on.json
        --micro-batch-size 2 --seed 1234 --learning-rate 0.001 --weight-decay 0.01)
case $mode in
    correctness) export RECSYS_OVERLAP_INPUT_AUDIT=1; flags=(--steps 8 --log-interval 4) ;;
    steady) flags=(--steps 60 --log-interval 10 --steady-warmup-steps 10 --steady-measure-steps 50) ;;
    level0) flags=(--steps 8 --log-interval 4 --profile --profile-active 3) ;;
esac
mkdir -- "$out"
printf 'bundle=%s\nmode=%s\norder=%s\nTASK_QUEUE_ENABLE=%s\nRECSYS_NPU_IDS=%s\n' \
    "$fingerprint" "$mode" "$order" "${TASK_QUEUE_ENABLE-<unset>}" "$RECSYS_NPU_IDS" > "$out/launch.txt"
for index in 0 1; do
    leg=${order:index:1}
    leg=${leg^^}
    printf -v name '%02d-%s' "$((index + 1))" "$leg"
    mkdir -- "$out/$name"
    (
        cd -- "$bundle/$leg"
        bash ./run_training.sh "${common[@]}" "${flags[@]}" --output-dir "$out/$name"
    ) > "$out/$name/run.log" 2>&1 || {
        failed=$?
        printf '%s\n' "$failed" > "$out/$name/train.exit"
        printf '%s failed, see %s\n' "$name" "$out/$name/run.log" >&2
        exit "$failed"
    }
    printf '0\n' > "$out/$name/train.exit"
done
exec python3 -B compare26.py "$mode" "$out" "$fingerprint" "${compare_arguments[@]}"
