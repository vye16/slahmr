#!/bin/bash

# for a shot change video, try --youtube_id "2VxpRal7wJE" 

python demo_online.py \
--track_dataset      "youtube" \
--youtube_id         "xEH_5T9jMVU" \
--storage_folder     "Videos_v1" \
--predict            "TPL" \
--distance_type      "EQ_010" \
--encode_type        "4c" \
--detect_shots       True \
--all_videos         True \
--track_history      7 \
--past_lookback      1 \
--max_age_track      50 \
--n_init             5 \
--low_th_c           0.8 \
--alpha              0.1 \
--hungarian_th       100 \
--render_type        "HUMAN_FULL_FAST" \
--render             True \
--res                256 \
--render_up_scale    2 \
--verbose            False \
--overwrite          True \
--use_gt             False \
--batch_id           -1 \
--detection_type     "mask" \
--start_frame        -1 \
--end_frame          100
