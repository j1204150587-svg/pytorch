#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
for argument in "$@"; do
  if [[ "$argument" == --help || "$argument" == -h || "$argument" == --dry-run ]]; then
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    exec python3 train_deepspeed.py "$@"
  fi
done
if [[ -z ${RECSYS_NPU_IDS:-} || ${RECSYS_NPU_IDS:-} == none ]]; then
  printf 'No NPU devices were selected for this container.\n' >&2
  exit 2
fi
set +u
source ./env.sh
set -u
# Docker has already restricted /dev/davinci* to the selected devices. CANN
# numbers this restricted list locally; a second filter using host IDs can hide it.
unset ASCEND_RT_VISIBLE_DEVICES
IFS=, read -ra selected_ids <<< "$RECSYS_NPU_IDS"
exec torchrun --standalone --nnodes=1 --nproc-per-node="${#selected_ids[@]}" train_deepspeed.py "$@"
