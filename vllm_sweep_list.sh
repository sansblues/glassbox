#!/usr/bin/env bash
# vllm_check.py configs from a list file, one fresh engine per line through vrun.sh (lock, gate, cap, timeout).
# Usage: [PY=...] [OUT=results/vllm_main] ./vllm_sweep_list.sh <list-file> [name-regex]
# Lines: name|ckpt|family|ms|n|extra|engine-json; extra = space-separated VAR=val env settings and/or script args
# (e.g. "VLLM_USE_SIMPLE_KV_OFFLOAD=1 --passes 2").
cd "$(dirname "$0")"
PY=${PY:-$HOME/.cache/claude-exp/vllm-main/bin/python}; OUT=${OUT:-results/vllm_main}
LIST=$1; SEL=${2:-.}
while IFS='|' read -r name ck family ms n extra engine; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  [[ "$name" =~ $SEL ]] || continue
  echo "=== $name ($ck $family ms=$ms n=$n $extra) $engine"
  envs=(); args=()
  for x in $extra; do [[ "$x" == *=* && "$x" != --* ]] && envs+=("$x") || args+=("$x"); done
  env PY=$PY SCRIPT=${SCRIPT:-vllm_check.py} "${envs[@]}" ./vrun.sh --out "$OUT" --ckpt "$ck" --name "$name" \
    --family "$family" --ms "$ms" --n "$n" "${args[@]}" --engine "$engine" < /dev/null 2>&1 | grep -v -e '^log:' -e 'Tokenizer not available' -e 'Confidential Computing' \
    | sed -n '/^RESULT/p;/^rc=/p;/WARNING/p;/GATE/p;/rror/p' | cut -c1-1200
done < "$LIST"
