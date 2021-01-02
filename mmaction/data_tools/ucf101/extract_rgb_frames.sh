#! /usr/bin/bash env

cd ../
python build_rawframes.py ../data/ucf101/videos/ ../data/ucf101/rawframes/ --level 2 --ext avi
echo "Raw frames RGB Generated"
cd ucf101/
