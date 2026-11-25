| config | family | attn | blk | kv | cudagraph | exact | per m | min margin | preempt | pc hits | gen s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 2 | 0 | 70.6 |
| eager | random | FLASH_ATTN | 64 | auto | NONE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 0 | 0 | 77.8 |
| chunk256 | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 0 | 0 | 86.5 |
| preempt | random | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 256/256 | 512:128/128 2048:128/128 | 38.50 | 177 | 0 | 525.1 |
| shared | shared | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 0 | 495808 | 100.6 |
| relabel | relabel | FLASH_ATTN | 64 | auto | FULL_AND_PIECEWISE | 384/384 | 64:128/128 512:128/128 2048:128/128 | 38.50 | 0 | 201152 | 139.5 |
