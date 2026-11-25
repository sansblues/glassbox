#!/usr/bin/env bash
# Historical-bug replay: vllm-project/vllm#47724, fixed by #47725 (392d1b4d, 2026-07-07). set_adapter_mapping memoized
# on the batch's LoRA-id mapping only, so an out-of-band add_lora() that LRU-evicts and re-activates the running
# adapters into swapped GPU slots leaves the punica routing stale: in-flight requests decode with the other
# adapter's weights. Pre-fix = parent 21b396ab (adjacent commits).
# Glass-box: A1 = oflip (flip applied at every step -> expected_vflip), A2 = zero (-> expected), max_loras=2;
# churn = add_lora(A3), add_lora(A4) after every engine step. Stale routing = antipodal pointer tokens.
# Model Runner V1 (VLLM_USE_V2_MODEL_RUNNER=0): MRV2 of this era refuses to start under WSL (no pinned memory); the
# bug is in vllm/lora/model_manager.py, shared by both runners.
# torchcodec was uninstalled from both venvs: it raises RuntimeError (no FFmpeg) where vLLM only catches ImportError.
cd "$(dirname "$0")"
run() {  # name venv phases engine-json
  echo "=== $1 ($2) phases=$3 $4"
  PY=$HOME/.cache/claude-exp/$2/bin/python SCRIPT=vllm_lora_churn.py VLLM_USE_V2_MODEL_RUNNER=${V2:-0} VLLM_USE_FLASHINFER_SAMPLER=0 \
    ./vrun.sh --ckpt "${CK:-ckpt_llama}" --name "$1" --phases "$3" --ms 64,512 --n 128 --engine "$4" \
    | sed -n '/^RESULT/p;/^rc=/p' | cut -c1-900
}
run bug47725_pre vllm-47725pre ref,churn,ref '{}'
run bug47725_fix vllm-47725fix ref,churn,ref '{}'
