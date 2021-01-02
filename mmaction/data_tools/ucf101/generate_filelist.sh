#! /usr/bin/bash env

cd ../../
PYTHONPATH=. python data_tools/build_file_list.py ucf101 data/ucf101/rawframes/ --level 2 --format rawframes --shuffle --num_split 3
echo "Filelist for rawframes generated."

PYTHONPATH=. python data_tools/build_file_list.py ucf101 data/ucf101/videos/ --level 2 --format videos --shuffle --num_split 3
echo "Filelist for videos generated."

cd data_tools/ucf101/
