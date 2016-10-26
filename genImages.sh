#!/bin/bash

#create images directory if not present
if [[ ! -d images ]]; then
    mkdir images
fi

#generate images
ffmpeg -ss  0 -i videos/Course\ B\ Cam1.mp4 -r 0.05 images/cb1_%03d.jpg
ffmpeg -ss  5 -i videos/Course\ B\ Cam2.mp4 -r 0.05 images/cb2_%03d.jpg
ffmpeg -ss 10 -i videos/Course\ B\ Cam3.mp4 -r 0.05 images/cb3_%03d.jpg
ffmpeg -ss 15 -i videos/Course\ B\ Cam4.mp4 -r 0.05 images/cb4_%03d.jpg
ffmpeg -ss  0 -i videos/Not\ Sure\ Cam1.mp4 -r 0.05 images/ns1_%03d.jpg
ffmpeg -ss  5 -i videos/Not\ Sure\ Cam2.mp4 -r 0.05 images/ns2_%03d.jpg
ffmpeg -ss 10 -i videos/Not\ Sure\ Cam3.mp4 -r 0.05 images/ns3_%03d.jpg
ffmpeg -ss 15 -i videos/Not\ Sure\ Cam4.mp4 -r 0.05 images/ns4_%03d.jpg

#rename images to 001.jpg, 002.jpg, ...
cd images
ls | { C=1; while read; do mv "$REPLY" $(printf '%03d.jpg' $C); C=$((C+1)); done }
cd ..
