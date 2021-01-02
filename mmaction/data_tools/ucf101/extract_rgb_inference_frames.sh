#! /usr/bin/bash env

cd ../
python build_rawframes.py ../data/ucf101/inference_video/ ../data/ucf101/rawframes/ --level 2 --ext avi
echo "Raw frames RGB Generated"
cd ucf101/
