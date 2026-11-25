| ckpt | config | kind | exact | min margin | alarm seqs | detail | metrics | gen s |
|---|---|---|---|---|---|---|---|---|
| ckpt_llama | smoke | random | 32/32 | 38.50 | 0 | 64:32/32 |  | 1.6 |
| ckpt_llama | default | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | preempt=1 | 58.1 |
| ckpt_llama | eager | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 |  | 64.6 |
| ckpt_llama | nopc | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | preempt=1 | 125.6 |
| ckpt_llama | chunk256 | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 |  | 80.7 |
| ckpt_llama | block16 | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 |  | 60.7 |
| ckpt_llama | smoke_abort | abort | 27/27 | 39.00 | 0 | orig:8/8 dup:9/9 resume:10/10 aborted-prefix-ok 11/11 |  | 0.9 |
| ckpt_llama | block32 | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | preempt=1 | 57.1 |
| ckpt_llama | smoke_stops | stops | 16/16 | 39.00 | 0 | stop:9/9 maxtok:7/7 | pc_hits=960 | 0.7 |
| ckpt_llama | block128 | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | preempt=1 | 65.2 |
| ckpt_llama | smoke_prio | priority | 16/16 | 39.00 | 0 | low:8/8 high:8/8 | pc_hits=960 | 4.0 |
| ckpt_llama | triton | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | preempt=1 | 132.5 |
| ckpt_llama | flex | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | preempt=5 | 191.5 |
| ckpt_llama | smoke_plp | plp | 16/16 | 39.00 | 0 | nolp_p1:2/2 plp_p1:6/6 nolp_p2:2/2 plp_p2:6/6 | prompt rank1 1560/1560 | pc_hits=320 | 6.9 |
| ckpt_llama | hf_impl | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | preempt=1 | 130.9 |
| ckpt_llama | smoke_lora | lora_mix | 32/32 | 39.00 | 0 | oflip_p1:4/4 zero_p1:4/4 base_p1:4/4 vflip_p1:4/4 oflip_p2:4/4 zero_p2:4/4 base_p2:4/4 vflip_p2:4/4 | pc_hits=2048 | 25.0 |
| ckpt_llama | seqs8 | random | 256/256 | 38.50 | 0 | 64:128/128 512:128/128 |  | 209.3 |
| ckpt_llama | smoke_sleep | sleep | 32/32 | 39.00 | 0 | pass1:16/16 pass2:16/16 | preempt=16 pc_hits=3008 | 8.3 |
| ckpt_llama | seqs256 | random | 768/768 | 38.50 | 0 | 64:256/256 512:256/256 2048:256/256 | preempt=121 | 253.5 |
| ckpt_llama | abort_pc | abort | 206/206 | 38.50 | 0 | orig:76/76 dup:67/67 resume:63/63 aborted-prefix-ok 78/78 | pc_hits=198912 | 33.0 |
| ckpt_llama | noasync | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | preempt=1 | 146.2 |
| ckpt_llama | abort_full | abort | 160/160 | 38.50 | 0 | orig:72/72 dup:60/60 resume:28/28 aborted-prefix-ok 74/74 | preempt=21 pc_hits=99008 acc=49876 drafts=196736 | 32.4 |
| ckpt_llama | preempt | random | 256/256 | 38.50 | 0 | 512:128/128 2048:128/128 | preempt=177 | 690.3 |
| ckpt_llama | abort_relabel_full | abort | 148/148 | 39.00 | 0 | orig:74/74 dup:57/57 resume:17/17 aborted-prefix-ok 72/72 | preempt=39 pc_hits=39488 acc=44368 drafts=177652 | 37.9 |
| ckpt_llama | preempt_nopc | random | 256/256 | 38.50 | 0 | 512:128/128 2048:128/128 | preempt=175 | 851.9 |
| ckpt_llama | abort_noasync_full | abort | 160/160 | 38.50 | 0 | orig:72/72 dup:60/60 resume:28/28 aborted-prefix-ok 74/74 | preempt=21 pc_hits=99008 acc=49875 drafts=196728 | 36.3 |
| ckpt_llama | preempt_chunk | random | 256/256 | 38.50 | 0 | 512:128/128 2048:128/128 | preempt=177 | 313.6 |
| ckpt_llama | stops_pc | stops | 192/192 | 38.50 | 0 | stop:107/107 maxtok:85/85 | preempt=30 pc_hits=79040 | 11.0 |
| ckpt_llama | mix | mix | 384/384 | 38.50 | 0 | 512:128/128 64:128/128 2048:128/128 |  | 71.5 |
| ckpt_llama | stops_full | stops | 192/192 | 38.50 | 0 | stop:107/107 maxtok:85/85 | pc_hits=79040 acc=25954 drafts=101672 | 6.3 |
| ckpt_llama | prio_preempt | priority | 128/128 | 38.50 | 0 | low:64/64 high:64/64 | preempt=138 pc_hits=52416 | 32.8 |
| ckpt_llama | prio_full | priority | 128/128 | 38.50 | 0 | low:64/64 high:64/64 | preempt=84 pc_hits=52352 acc=37092 drafts=146264 | 28.2 |
| ckpt_llama | nfork_pc | nfork | 512/512 | 38.50 | 0 | fork_p1:256/256 fork_p2:256/256 | pc_hits=283840 | 46.4 |
| ckpt_llama | nfork_full | nfork | 512/512 | 38.50 | 0 | fork_p1:256/256 fork_p2:256/256 | preempt=218 pc_hits=273536 acc=147924 drafts=583676 | 21.1 |
| ckpt_llama | plp_pc | plp | 128/128 | 38.50 | 0 | nolp_p1:33/33 plp_p1:31/31 nolp_p2:33/33 plp_p2:31/31 | prompt rank1 34940/34940 | pc_hits=32704 | 9.0 |
| ckpt_llama | shared | shared | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | pc_hits=495808 | 50.5 |
| ckpt_llama | plp_chunk | plp | 128/128 | 38.50 | 0 | nolp_p1:33/33 plp_p1:31/31 nolp_p2:33/33 plp_p2:31/31 | prompt rank1 34940/34940 | preempt=28 pc_hits=27776 | 12.1 |
| ckpt_llama | shared_nopc | shared | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | preempt=1 | 61.6 |
| ckpt_llama | plp_start_cached | plp | 128/128 | 38.50 | 0 | nolp_p1:33/33 plp_p1:31/31 nolp_p2:33/33 plp_p2:31/31 | prompt rank1 17694/17694 | pc_hits=47168 | 14.4 |
| ckpt_llama | shared_chunk | shared | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | pc_hits=495808 | 55.0 |
| ckpt_llama | plp_start_cached_chunk | plp | 128/128 | 38.50 | 0 | nolp_p1:33/33 plp_p1:31/31 nolp_p2:33/33 plp_p2:31/31 | prompt rank1 17694/17694 | preempt=22 pc_hits=43456 | 10.2 |
| ckpt_llama | shared_block16 | shared | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | pc_hits=499872 | 51.9 |
| ckpt_llama | lora_mix2 | lora_mix | 256/256 | 38.50 | 0 | vflip_p1:32/32 oflip_p1:32/32 base_p1:32/32 zero_p1:32/32 vflip_p2:32/32 oflip_p2:32/32 base_p2:32/32 zero_p2:32/32 | pc_hits=73728 | 13.8 |
| ckpt_llama | shared_preempt | shared | 256/256 | 38.50 | 0 | 512:128/128 2048:128/128 | preempt=392 pc_hits=487680 | 81.2 |
| ckpt_llama | lora_mix1 | lora_mix | 256/256 | 38.50 | 0 | vflip_p1:32/32 oflip_p1:32/32 base_p1:32/32 zero_p1:32/32 vflip_p2:32/32 oflip_p2:32/32 base_p2:32/32 zero_p2:32/32 | pc_hits=73728 | 19.1 |
| ckpt_llama | relabel | relabel | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | pc_hits=201152 | 54.0 |
| ckpt_llama | lora_mix_full | lora_mix | 256/256 | 38.50 | 0 | vflip_p1:32/32 oflip_p1:32/32 base_p1:32/32 zero_p1:32/32 vflip_p2:32/32 oflip_p2:32/32 base_p2:32/32 zero_p2:32/32 | preempt=114 | 31.5 |
| ckpt_llama | relabel_chunk | relabel | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | pc_hits=201152 | 67.6 |
| ckpt_llama | sleep_l1_keep | sleep | 192/192 | 38.50 | 0 | pass1:96/96 pass2:96/96 | preempt=96 pc_hits=94400 | 11.6 |
| ckpt_llama | relabel_block16 | relabel | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | pc_hits=209232 | 56.9 |
| ckpt_llama | relabel_preempt | relabel | 256/256 | 38.50 | 0 | 512:128/128 2048:128/128 | preempt=237 pc_hits=224192 | 110.5 |
| ckpt_llama | sleep_l2_keep | sleep | 192/192 | 38.50 | 0 | pass1:96/96 pass2:96/96 | preempt=96 pc_hits=94400 | 6.8 |
| ckpt_llama | long4096 | random | 48/48 | 38.50 | 0 | 4096:48/48 |  | 74.6 |
| ckpt_llama | sleep_l0_keep | sleep | 192/192 | 38.50 | 0 | pass1:96/96 pass2:96/96 | pc_hits=94400 | 9.6 |
| ckpt_llama | long4096_chunk | random | 32/32 | 38.50 | 0 | 4096:32/32 |  | 55.4 |
| ckpt_llama | ngram | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | preempt=1 acc=336237 drafts=1342036 | 60.1 |
| ckpt_llama | sleep_l1_keep_full | sleep | 192/192 | 38.50 | 0 | pass1:96/96 pass2:96/96 | preempt=87 pc_hits=82112 acc=55555 drafts=219840 | 10.8 |
| ckpt_llama | ngram_shared | shared | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | pc_hits=495808 acc=336460 drafts=1341180 | 53.4 |
| ckpt_llama | all5_shared | shared | 384/384 | 38.50 | 0 | p1:512:96/96 p1:2048:96/96 p2:512:96/96 p2:2048:96/96 | preempt=324 pc_hits=729600 acc=492306 drafts=1961400 | 157.6 |
| ckpt_llama | sleep_l1_abort_v2 | sleep | 192/192 | 38.50 | 0 | pass1:0/0 resent:96/96 pass2:96/96 aborted-prefix-ok 96/96 | pc_hits=133504 | 13.7 |
| ckpt_llama | ngram_relabel | relabel | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | pc_hits=201152 acc=336000 drafts=1342976 | 57.3 |
| ckpt_llama | all5_relabel | relabel | 384/384 | 38.50 | 0 | p1:512:96/96 p1:2048:96/96 p2:512:96/96 p2:2048:96/96 | preempt=240 pc_hits=297984 acc=491694 drafts=1964488 | 164.2 |
| ckpt_llama | ngram_chunk | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | acc=336238 drafts=1342036 | 83.6 |
| ckpt_llama | all5_mix | mix | 192/192 | 38.50 | 0 | 512:64/64 2048:64/64 64:64/64 | preempt=124 acc=168016 drafts=670948 | 70.0 |
| ckpt_llama | ngram_preempt | random | 256/256 | 38.50 | 0 | 512:128/128 2048:128/128 | preempt=236 acc=327739 drafts=1309544 | 128.4 |
| ckpt_llama | offload_shared | shared | 256/256 | 38.50 | 0 | p1:512:128/128 p2:512:128/128 | preempt=399 pc_hits=196096 ext_hits=32512 | 33.2 |
| ckpt_llama | ngram_noasync | random | 384/384 | 38.50 | 0 | 64:128/128 512:128/128 2048:128/128 | preempt=1 acc=336237 drafts=1342036 | 73.6 |
| ckpt_llama | offload_random | random | 256/256 | 38.50 | 0 | p1:512:128/128 p2:512:128/128 | preempt=184 ext_hits=131072 | 96.9 |
| ckpt_llama | rope1e4 | random | 94/256 **X** | 2.00 | 29 | 64:94/128 512:0/128 |  | 17.2 |
| ckpt_llama | offload_full | shared | 256/256 | 38.50 | 0 | p1:512:64/64 p1:2048:64/64 p2:512:64/64 p2:2048:64/64 | preempt=485 pc_hits=486912 ext_hits=82688 acc=327694 drafts=1308764 | 308.6 |
| ckpt_llama | abort_offload | abort | 207/207 | 38.50 | 0 | orig:75/75 dup:66/66 resume:66/66 aborted-prefix-ok 79/79 | preempt=211 pc_hits=187328 ext_hits=6656 kv_offload_size{transfer_type=CPU_to_GPU}=79822848.0 kv_offload_size{transfer_type=GPU_to_CPU}=60686336.0 kv_offload_load_size=79822848.0 kv_offload_store_size=60686336.0 | 36.9 |
| ckpt_llama | offload_simple_shared | shared | 256/256 | 38.50 | 0 | p1:512:128/128 p2:512:128/128 | preempt=371 pc_hits=196352 ext_hits=32256 | 47.9 |
| ckpt_llama | stops_offload | stops | 192/192 | 38.50 | 0 | stop:107/107 maxtok:85/85 | preempt=26 pc_hits=79040 kv_offload_size{transfer_type=CPU_to_GPU}=11010048.0 kv_offload_size{transfer_type=GPU_to_CPU}=64618496.0 kv_offload_load_size=11010048.0 kv_offload_store_size=64618496.0 | 6.6 |
