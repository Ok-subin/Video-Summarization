#!/bin/bash

'''
This script decompose a video into frames
How to use: replace path_to_videos and path_to_frames with real paths
'''

for f in video/*.mp4
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name\
  basename=$(ff=${f%.ext} ; echo ${ff##*/})
  name=$(echo $basename | cut -d'.' --complement -f2-)
  echo $f
 mkdir -p video/"$name"
 ffmpeg -i "$f" -f image2 video/video_frame/"$name"/%06d.jpg
 #ffmpeg -i "$f" -f image2 frames/"$name"/%6d.jpg
done


