#!/bin/bash

start_index=0
end_index=-1
gpu=0
month=2403
target_dir=/path/to/your/html/base/dir/${month}

CUDA_VISIBLE_DEVICES=$gpu nohup python -um connector.html_parsing \
--start_index $start_index \
--end_index $end_index \
--target_dir $target_dir > indexing_${gpu}.out_${month} 2> indexing_${gpu}.log_${month} &