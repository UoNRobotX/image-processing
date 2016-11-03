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
2. `./main.sh generate`

## Creating training/testing data.
* Label static filter: `./main.sh mark filter`
* Label coarse network data: `./main.sh mark coarse`
* Label detailed network data: `./main.sh mark detailed`

## Viewing the training/testing data.
* View filter: `./main.sh view filter`
* View coarse training data: `./main.sh view coarse`
* View detailed training data: `./main.sh view detailed`

## Working with the coarse network.
* Train the coarse network, for 100 steps: `./main.sh train coarse`
* Test the coarse network: `./main.sh test coarse`
* Run the coarse network on a random image: `./main.sh run coarse`
* Generate sample coarse network inputs: `./main.sh samples coarse`
* View graph and training/testing statistics with tensorboard:
    * The coarse should have been trained or tested at least once.
    * `tensorboard --logdir=coarseSummaries`
    * Open a browser to localhost:6006

## Working with the whole network.
* Train the detailed network, for 100 steps: `./main.sh train detailed`
* Test the detailed network: `./main.sh test detailed`
* Run the detailed network on a random image: `./main.sh run detailed`
* Generate sample detailed network inputs: `./main.sh samples detailed`
* View graph and training/testing statistics with tensorboard.
    * The detailed network should have been trained or tested at least once.
    * `tensorboard --logdir=detailedSummaries`
    * Open a browser to localhost:6006

