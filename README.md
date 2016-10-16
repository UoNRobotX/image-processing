# RobotX image processing stuff

## Dependencies
* Tensorflow
* Python 3
* Python 3 Pillow (for processing image files)
* Python 3 Tk (optional, used to provide a GUI for generating training/testing data)

## An example of generating training data from videos
1. Place videos in videos/, and cd to it.
2. Get images from each video, placing them in images/.
   `ffmpeg -ss  0 -i Course\ B\ Cam1.mp4 -r 0.05 ../images/cb1_%03d.jpg`
   `ffmpeg -ss  5 -i Course\ B\ Cam2.mp4 -r 0.05 ../images/cb2_%03d.jpg`
   `ffmpeg -ss 10 -i Course\ B\ Cam3.mp4 -r 0.05 ../images/cb3_%03d.jpg`
   `ffmpeg -ss 15 -i Course\ B\ Cam4.mp4 -r 0.05 ../images/cb4_%03d.jpg`
   `ffmpeg -ss  0 -i Not\ Sure\ Cam1.mp4 -r 0.05 ../images/ns1_%03d.jpg`
   `ffmpeg -ss  5 -i Not\ Sure\ Cam2.mp4 -r 0.05 ../images/ns2_%03d.jpg`
   `ffmpeg -ss 10 -i Not\ Sure\ Cam3.mp4 -r 0.05 ../images/ns3_%03d.jpg`
   `ffmpeg -ss 15 -i Not\ Sure\ Cam4.mp4 -r 0.05 ../images/ns4_%03d.jpg`
3. Rename the image files.
   `cd ../images`
   `ls | { C=1; while read; do mv "$REPLY" $(printf '%03d.jpg' $C); C=$((C+1)); done }`
4. `cd ..`
5. Label the grid cells to use for the filter (used to ignore camera boundaries, WAM-V 'roof', etc).
   `python3 markImages.py -f -d images > filterData.txt`
6. Label water grid cells (used to train/test coarse network).
   `python3 markImages.py -w -d images > data.txt`
7. Split the coarse network data into training and test sets.
   `csplit data.txt '/075\.jpg/' -s`
   `mv xx00 trainingDataCoarse.txt`
   `mv xx01 testingDataCoarse.txt`
   `rm data.txt`
6. Label buoy boxes.
   `python3 markImages.py -b -d images > data.txt`
7. Split the data into training and test sets.
   `csplit data.txt '/075\.jpg/' -s`
   `mv xx00 trainingData.txt`
   `mv xx01 testingData.txt`
   `rm data.txt`
8. View filter.
   `python3 showMarkedImages.py -f filterData.txt images`
9. View labelled water grid cells.
   `python3 showMarkedImages.py -w trainingDataCoarse.txt`
10. View labelled boxes.
   `python3 showMarkedImages.py -b trainingData.txt`

## Working with the coarse network.
* Creating and training the coarse network, for 100 steps.
  `python3 findBuoys.py -c -n -f filterData.txt -t trainingDataCoarse.txt -b 100`
* Testing the coarse network.
  `python3 findBuoys.py -c -f filterData.txt -e testingDataCoarse.txt`
* View the results of running the coarse network on an image.
  `python3 findBuoys.py -c -f filterData.txt -r images/008.jpg`
   A representation of the results is saved in outputFindBuoys.jpg.
* Viewing sample coarse network inputs.
  `python3 findBuoys.py -c -f filterData.txt -s trainingDataCoarse.txt`
  A representation of the results is saved in samplesFindBuoys.jpg.

## Working with the whole network.
* Creating and training the network, for 100 steps.
  `python3 findBuoys.py -n -f filterData.txt -t trainingData.txt -b 100`
* Testing the network.
  `python3 findBuoys.py -f filterData.txt -e testingData.txt`
* View the results of running the network on an image.
  `python3 findBuoys.py -f filterData.txt -r images/008.jpg`
   A representation of the results is saved in outputFindBuoys.jpg.
* Viewing sample network inputs.
  `python3 findBuoys.py -f filterData.txt -s trainingData.txt`
  A representation of the results is saved in samplesFindBuoys.jpg.
