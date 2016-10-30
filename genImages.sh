#!/bin/bash

#create images directory if not present
if [[ ! -d images ]]; then
    mkdir images
fi

#generate images
ffmpeg -ss  00:20 -i videos/Course\ B\ Cam1.mp4 -r 1 -t 01:17 -qscale 5 images/cb1_1_%03d.jpg
ffmpeg -ss  01:51 -i videos/Course\ B\ Cam1.mp4 -r 1 -t 00:10 -qscale 5 images/cb1_2_%03d.jpg
ffmpeg -ss  02:35 -i videos/Course\ B\ Cam1.mp4 -r 1 -t 00:10 -qscale 5 images/cb1_3_%03d.jpg
ffmpeg -ss  01:40 -i videos/Course\ B\ Cam2.mp4 -r 1 -t 00:20 -qscale 5 images/cb2_1_%03d.jpg
ffmpeg -ss  02:43 -i videos/Course\ B\ Cam2.mp4 -r 1 -t 00:11 -qscale 5 images/cb2_2_%03d.jpg
ffmpeg -ss  01:40 -i videos/Course\ B\ Cam3.mp4 -r 1 -t 00:20 -qscale 5 images/cb3_1_%03d.jpg
ffmpeg -ss  02:43 -i videos/Course\ B\ Cam3.mp4 -r 1 -t 00:11 -qscale 5 images/cb3_2_%03d.jpg
ffmpeg -ss  01:51 -i videos/Course\ B\ Cam4.mp4 -r 1 -t 00:10 -qscale 5 images/cb4_1_%03d.jpg
ffmpeg -ss  04:48 -i videos/Course\ B\ Cam4.mp4 -r 1 -t 00:21 -qscale 5 images/cb4_2_%03d.jpg

##remove some problematic images (results may vary depending on ffmpeg)
#rm 077.jpg {098..100}.jpg {103..106}.jpg {122..128}.jpg {166..171}.jpg 176.jpg

#rename images to 001.jpg, 002.jpg, ...
cd images
ls | { C=1; while read; do mv "$REPLY" $(printf '%03d.jpg' $C); C=$((C+1)); done }
cd ..
