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
3. Rename the image files (optional).
   `cd ../images`
   `ls | { C=1; while read; do mv "$REPLY" $(printf '%03d.jpg' $C); C=$((C+1)); done }`
4. Label buoys.
   `cd ..`
   `ls images/* | python3 genData.py > trainingData.csv`
5. View training data (optional).
   `python3 displayData.py trainingData.csv`

## Training a new network.
1. `python3 findBuoys.py -t trainingData.csv -n`

## Testing the network.
1. `python3 findBuoys.py -e testingData.csv`

## View the results of running the network on an image.
1. `python3 findBuoys.py -r images/008.jpg`
   A representation of the results is saved in outputFindBuoys.jpg.

## Viewing sample network inputs.
1. `python3 findBuoys.py -s trainingData.csv`
   A representation of the results is saved in samplesFindBuoys.jpg.
