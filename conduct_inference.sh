cd data
python video_preprocessing_script.py
cd ../data_tools/ucf101
bash extract_rgb_inference_frames.sh 
cd ../..
bash tools/dist_inference_recongizer.sh test_configs/TSN/ucf101/tsn_test.py work_dirs/tsn_2d_rgb_bninception_seg_3_f1s1_b32_g8/latest.pth 1 --use_softmax
cd data
python post_processing.py
