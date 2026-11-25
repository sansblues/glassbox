| config | family | attn | blk | kv | cudagraph | exact | per m | min margin | preempt | pc hits | gen s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eager | random | FLASH_ATTN | 64 | auto | NONE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 0 | 0 | 56.1 |
| nopc | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 2 | 0 | 55.5 |
| chunk256 | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 0 | 0 | 76.3 |
| block16 | random | FLASH_ATTN | 16 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 1 | 0 | 57.7 |
| block32 | random | FLASH_ATTN | 32 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 1 | 0 | 58.8 |
| block128 | random | FLASH_ATTN | 128 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 2 | 0 | 57.6 |
| triton | random | TRITON_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 2 | 0 | 51.0 |
| flex | random | FLEX_ATTENTION | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 5 | 0 | 105.1 |
| preempt | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 256/256 | 512:128/128 2048:128/128 | 38.50 | 177 | 0 | 219.8 |
| preempt_nopc | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 256/256 | 512:128/128 2048:128/128 | 38.50 | 177 | 0 | 254.2 |
| preempt_chunk | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 256/256 | 512:128/128 2048:128/128 | 38.50 | 177 | 0 | 199.1 |
| mix | mix | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 512:128/128 64:128/128 2048:128/128 | 38.50 | 0 | 0 | 63.3 |
| shared | shared | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 0 | 495808 | 52.8 |
| shared_nopc | shared | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 2 | 0 | 56.4 |
| shared_chunk | shared | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 0 | 495808 | 53.3 |
| shared_preempt | shared | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 256/256 | 512:128/128 2048:128/128 | 38.50 | 388 | 487680 | 107.5 |
| relabel | relabel | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 0 | 201152 | 53.1 |
| relabel_chunk | relabel | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 0 | 201152 | 75.7 |
| relabel_preempt | relabel | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 256/256 | 512:128/128 2048:128/128 | 38.50 | 231 | 224192 | 151.6 |
| long4096 | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 128/128 | 4096:128/128 | 38.50 | 62 | 0 | 265.7 |
| long4096_chunk | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 64/64 | 4096:64/64 | 38.50 | 0 | 0 | 137.9 |
| default | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 2 | 0 | 86.0 |
| hf_impl | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 1 | 0 | 55.6 |
| seqs8 | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 256/256 | 64:128/128 512:128/128 | 38.50 | 0 | 0 | 60.8 |
| seqs256 | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 768/768 | 64:256/256 512:256/256 2048:256/256 | 38.50 | 121 | 0 | 149.2 |
| noasync | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 2 | 0 | 85.2 |
| shared_block16 | shared | FLASH_ATTN | 16 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 0 | 499872 | 58.7 |
| relabel_block16 | relabel | FLASH_ATTN | 16 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 0 | 209232 | 64.8 |
| bug64279_pre_hf_mode2 | random | ? | 16 | auto | FULL_DECODE_ONLY | 0/256 | 64:0/128 512:0/128 | 0.00 | 0 | 0 | 29.9 |
| bug64279_fix_hf_mode2 | random | ? | 16 | auto | FULL_DECODE_ONLY | 256/256 | 64:128/128 512:128/128 | 38.50 | 0 | 0 | 28.2 |
| bug64279_pre_native_mode2 | random | ? | 16 | auto | FULL_DECODE_ONLY | 256/256 | 64:128/128 512:128/128 | 38.50 | 0 | 0 | 42.8 |
| bug64279_pre_hf_default | random | ? | 16 | auto | FULL_AND_PIECEWISE | 256/256 | 64:128/128 512:128/128 | 38.50 | 0 | 0 | 29.9 |
| rope1e4 | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 94/256 | 64:94/128 512:0/128 | 2.00 | 0 | 0 | 29.4 |
