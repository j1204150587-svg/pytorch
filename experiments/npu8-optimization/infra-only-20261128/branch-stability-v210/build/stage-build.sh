#!/bin/bash
set -euo pipefail
base=/data/recsys-gr-ds019-host88/artifacts/pta-build-UK5IES/checkout
out=/data/recsys-gr-ds019-host88/artifacts/infra-only-20261128/branch-stability-v210/build
src="$out/cache-worktree"
test ! -e "$src"
cp -a "$base" "$src"
cd "$src"
git -c safe.directory='*' bundle verify "$out/branch.bundle"
git -c safe.directory='*' fetch "$out/branch.bundle" refs/heads/codex/hstu-dynamic-eager-infra-v2.10
git -c safe.directory='*' restore --source=HEAD --staged --worktree -- torch_npu/csrc/core/npu/NPUQueue.cpp torch_npu/csrc/framework/OpCommand.cpp
git -c safe.directory='*' switch --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = e3cda6399cb194c813530222478ea018f5aca5c9
test "$(git -C third_party/op-plugin rev-parse HEAD)" = eaf6b55812a1998a4a40e485e3b2861a40e9c2f6
# The generator had modified only test lists. These files are private to our new copy.
git -C third_party/op-plugin restore --source=HEAD -- test
git status --short --untracked-files=no > "$out/checkout-status.txt"
git submodule status > "$out/submodules.txt"
git -C third_party/op-plugin status --short --untracked-files=no > "$out/op-plugin-status.txt"
test ! -s "$out/op-plugin-status.txt"
git log --format='%H %P %s' --reverse 38e483ee79ec8a3286517808740a811c0f0ae66c..HEAD > "$out/commits.txt"
img=sha256:c342a038f6476c794fb364d96479286ebd456021e86dba0d3b715ce4cb983e33
docker run --name recsys-dynbranch-e3cda63-build --runtime runc --network none --cpus 8 --memory 32g --entrypoint /bin/bash -v "$src:/artifacts/pta-build-UK5IES/checkout" -v "$out:/result" "$img" /result/configure-build.sh
