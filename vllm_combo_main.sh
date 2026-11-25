#!/usr/bin/env bash
# Lifecycle / feature-combination matrix (vllm_combo.py), one fresh engine per line through vrun.sh (GPU lock,
# memory gate, cap, timeout).  Usage: [PY=...] [OUT=results/vllm_main] ./vllm_combo_main.sh <list-file> [name-regex]
# List lines: name|ckpt|scenario|family|ms|n|params-json|engine-json   (# = comment)
cd "$(dirname "$0")"
PY=${PY:-$HOME/.cache/claude-exp/vllm-main/bin/python}; OUT=${OUT:-results/vllm_main}
LIST=$1; SEL=${2:-.}
while IFS='|' read -r name ck scen family ms n params engine; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  [[ "$name" =~ $SEL ]] || continue
  echo "=== $name ($ck $scen $family ms=$ms n=$n) params=$params engine=$engine"
  PY=$PY SCRIPT=vllm_combo.py ./vrun.sh --out "$OUT" --ckpt "$ck" --name "$name" --scenario "$scen" --family "$family" \
    --ms "$ms" --n "$n" --params "$params" --engine "$engine" < /dev/null 2>&1 | grep -v -e '^log:' -e 'Tokenizer not available' -e 'Confidential Computing' \
    | sed -n '/^RESULT/p;/^rc=/p;/GATE/p;/rror/p' | cut -c1-1500
done < "$LIST"
