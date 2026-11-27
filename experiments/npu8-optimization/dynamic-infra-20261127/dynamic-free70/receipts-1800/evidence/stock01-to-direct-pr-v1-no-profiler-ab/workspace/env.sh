#!/usr/bin/env bash
# docker exec starts a fresh environment. Use the image's selected CANN installation.
_recsys_load_cann() {
  local script root ir
  local prefix=${ASCEND_INSTALL_PATH:-/usr/local/Ascend}
  local -a main_scripts=()
  if [[ -n ${CANN_ENV_SCRIPT:-} ]]; then
    main_scripts=("$CANN_ENV_SCRIPT")
  else
    if [[ -n ${ASCEND_HOME_PATH:-} ]]; then
      main_scripts+=("$ASCEND_HOME_PATH/set_env.sh")
    fi
    main_scripts+=("$prefix/ascend-toolkit/set_env.sh" "$prefix/cann/set_env.sh"
                   "$prefix/set_env.sh" "$prefix"/cann-*/set_env.sh)
  fi
  for script in "${main_scripts[@]}"; do
    [[ -f "$script" ]] || continue
    source "$script" || return
    root=$(dirname "$(readlink -f "$script")")
    # Some versions ship a separate IR environment, others set it up in the main script.
    for ir in "${ASCEND_HOME_PATH:-$root}/share/info/ascendnpu-ir/bin/set_env.sh" \
              "$root/share/info/ascendnpu-ir/bin/set_env.sh" \
              "$prefix/cann/share/info/ascendnpu-ir/bin/set_env.sh"; do
      if [[ -f "$ir" ]]; then
        source "$ir" || return
        break
      fi
    done
    return 0
  done
  printf 'CANN setup script unavailable. Set CANN_ENV_SCRIPT to the installed set_env.sh.\n' >&2
  return 1
}
_recsys_load_cann || return 1
unset -f _recsys_load_cann
export DS_ACCELERATOR=npu
export DS_BUILD_OPS=0
export PYTHONUNBUFFERED=1
