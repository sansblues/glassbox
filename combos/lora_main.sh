#!/usr/bin/env bash
# LoRA same-name / distinct-name phases (vllm_lora_check.py) on main, Llama + Qwen3
cd "$(dirname "$0")/.."
run() { echo "=== $1 ($2) phases=$3 $4"; PY=$HOME/.cache/claude-exp/vllm-main/bin/python SCRIPT=vllm_lora_check.py \
  ./vrun.sh --out results/vllm_main --ckpt "$2" --name "$1" --phases "$3" --ms 64,512 --n 128 --engine "$4" < /dev/null 2>&1 \
  | sed -n '/^RESULT/p;/^rc=/p;/GATE/p' | cut -c1-700; }
run lora_samename ckpt_llama vflip:glass,zero:glass '{}'
run lora_samename_rev ckpt_llama zero:glass,vflip:glass '{}'
run lora_distinct ckpt_llama vflip:glass_v,zero:glass_z '{}'
run lora_oflip_base ckpt_llama oflip:glass_o,base:none,vflip:glass_v '{}'
run lora_samename ckpt_qwen3 vflip:glass,zero:glass '{}'
run lora_distinct ckpt_qwen3 vflip:glass_v,zero:glass_z '{}'
