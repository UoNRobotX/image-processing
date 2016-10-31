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

## Generating the images.
1. Create a videos/ directory, and place the videos in it.
2. `./genImages.sh`

## Creating training/testing data.
1. Label the grid cells to use for the filter (used to ignore camera boundaries, WAM-V 'roof', etc).
    * `python3 markImages.py filter -d images -o filterData.txt`
2. Label water grid cells (used to train/test coarse network).
    * `python3 markImages.py coarse -d images -o data.txt`
3. Split the coarse network data into training and test sets.
    * `csplit data.txt '/125\.jpg/' -s`
    * `mv xx00 trainingDataCoarse.txt`
    * `mv xx01 testingDataCoarse.txt`
    * `rm data.txt`
4. Label buoy boxes.
    * `python3 markImages.py detailed -d images -o data.txt`
5. Split the data into training and test sets.
    * `csplit data.txt '/125\.jpg/' -s`
    * `mv xx00 trainingData.txt`
    * `mv xx01 testingData.txt`
    * `rm data.txt`

## Viewing the training/testing data.
* View filter.
    * `python3 markImages.py -f -d images -l filterData.txt >/dev/null`
* View labelled water grid cells.
    * `python3 markImages.py -w -d images -l trainingDataCoarse.txt >/dev/null`
* View labelled boxes.
    * `python3 markImages.py -b -d images -l trainingData.txt >/dev/null`

## Working with the coarse network.
* Train the coarse network, for 100 steps.
    * `python3 findBuoys.py train trainingDataCoarse.txt filterData.txt -c -s 100`
* Test the coarse network.
    * `python3 findBuoys.py test testingDataCoarse.txt filterData.txt -c`
* View the results of running the coarse network on an image. After running, a representation of the results is saved in outputFindBuoys.jpg.
    * `python3 findBuoys.py run images/008.jpg filterData.txt -c`
* View sample coarse network inputs. After running, a representation of the results is saved in samplesFindBuoys.jpg.
    * `python3 findBuoys.py samples trainingDataCoarse.txt filterData.txt -c`

## Working with the whole network.
* Train the network, for 100 steps.
    * `python3 findBuoys.py train trainingData.txt filterData.txt -s 100`
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
