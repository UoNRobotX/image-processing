# RobotX image processing stuff

## Dependencies
* Tensorflow
* Python 3
* Python 3 Pillow (for processing image files)
* Python 3 Tk (optional, used to provide a GUI for generating training/testing data)

## An example of installing the dependencies on Debian 8.4.
* `apt-get update`
* `apt-get install python3-dev python3-pip`
* `export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc1-cp35-cp35m-linux_x86_64.whl`
* `pip3 install --upgrade $TF_BINARY_URL`
* `apt-get install libtiff5-dev libjpeg-dev zlib1g-dev libfreetype6-dev`
* `apt-get install liblcms2-dev libwebp-dev tcl-dev tk-dev`
* `pip3 install Pillow`
* `apt-get install python3-tk`

## An example of generating training data from videos
1. Place videos in videos/, and cd to it.
2. Get images from each video, placing them in images/.
    * `ffmpeg -ss  0 -i Course\ B\ Cam1.mp4 -r 0.05 ../images/cb1_%03d.jpg`
    * `ffmpeg -ss  5 -i Course\ B\ Cam2.mp4 -r 0.05 ../images/cb2_%03d.jpg`
    * `ffmpeg -ss 10 -i Course\ B\ Cam3.mp4 -r 0.05 ../images/cb3_%03d.jpg`
    * `ffmpeg -ss 15 -i Course\ B\ Cam4.mp4 -r 0.05 ../images/cb4_%03d.jpg`
    * `ffmpeg -ss  0 -i Not\ Sure\ Cam1.mp4 -r 0.05 ../images/ns1_%03d.jpg`
    * `ffmpeg -ss  5 -i Not\ Sure\ Cam2.mp4 -r 0.05 ../images/ns2_%03d.jpg`
    * `ffmpeg -ss 10 -i Not\ Sure\ Cam3.mp4 -r 0.05 ../images/ns3_%03d.jpg`
    * `ffmpeg -ss 15 -i Not\ Sure\ Cam4.mp4 -r 0.05 ../images/ns4_%03d.jpg`
3. Rename the image files.
    * `cd ../images`
    * `ls | { C=1; while read; do mv "$REPLY" $(printf '%03d.jpg' $C); C=$((C+1)); done }`
4. `cd ..`
5. Label the grid cells to use for the filter (used to ignore camera boundaries, WAM-V 'roof', etc).
    * `python3 markImages.py -f -d images > filterData.txt`
6. Label water grid cells (used to train/test coarse network).
    * `python3 markImages.py -w -d images > data.txt`
7. Split the coarse network data into training and test sets.
    * `csplit data.txt '/075\.jpg/' -s`
    * `mv xx00 trainingDataCoarse.txt`
    * `mv xx01 testingDataCoarse.txt`
    * `rm data.txt`
6. Label buoy boxes.
    * `python3 markImages.py -b -d images > data.txt`
7. Split the data into training and test sets.
    * `csplit data.txt '/075\.jpg/' -s`
    * `mv xx00 trainingData.txt`
    * `mv xx01 testingData.txt`
    * `rm data.txt`
8. View filter.
    * `python3 showMarkedImages.py -f filterData.txt images`
9. View labelled water grid cells.
    * `python3 showMarkedImages.py -w trainingDataCoarse.txt`
10. View labelled boxes.
    * `python3 showMarkedImages.py -b trainingData.txt`

## Working with the coarse network.
* Create and train the coarse network, for 100 steps.
    * `python3 findBuoys.py train trainingDataCoarse.txt filterData.txt -c -n -s 100`
* Test the coarse network.
    * `python3 findBuoys.py test testingDataCoarse.txt filterData.txt -c`
* View the results of running the coarse network on an image. After running, a representation of the results is saved in outputFindBuoys.jpg.
    * `python3 findBuoys.py run images/008.jpg filterData.txt -c`
* View sample coarse network inputs. After running, a representation of the results is saved in samplesFindBuoys.jpg.
    * `python3 findBuoys.py samples trainingDataCoarse.txt filterData.txt -c`

## Working with the whole network.
* Create and train the network, for 100 steps.
    * `python3 findBuoys.py train trainingData.txt filterData.txt -n -s 100`
* Test the network.
    * `python3 findBuoys.py test testingData.txt filterData.txt`
* View the results of running the network on an image. After running, a representation of the results is saved in outputFindBuoys.jpg.
    * `python3 findBuoys.py run images/008.jpg filterData.txt`
* View sample network inputs. After running, a representation of the results is saved in samplesFindBuoys.jpg.
    * `python3 findBuoys.py samples trainingData.txt filterData.txt`

## Viewing graph and training/testing statistics with tensorboard.
* The coarse or detailed network should have been trained or tested at least once.
* `tensorboard --logdir=summaries`
* Open a browser to localhost:6006
