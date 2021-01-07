cd data
python video_preprocessing_script.py 
cd ../data_tools/ucf101/
bash extract_rgb_frames.sh
bash generate_filelist.sh
cd ../..
