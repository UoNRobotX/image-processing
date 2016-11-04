#!/bin/bash

set -e

USAGE="Usage:
    $0 generate
    $0 mark filter|coarse|detailed
    $0 view filter|coarse|detailed
    $0 train coarse|detailed
    $0 test coarse|detailed
    $0 run coarse|detailed
    $0 samples coarse|detailed
"

CMD="$1"
SUBCMD="$2"
shift 2 || true

case "$CMD" in
    "generate")
        #create images directory if absent
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
        #remove some problematic images (results may vary depending on ffmpeg)
        rm 077.jpg {098..100}.jpg {103..106}.jpg {122..128}.jpg {166..171}.jpg 176.jpg
        #rename images to 001.jpg, 002.jpg, ...
        cd images
        ls | { C=1; while read; do mv "$REPLY" $(printf '%03d.jpg' $C); C=$((C+1)); done }
        cd ..
    ;;
    "mark")
        case "$SUBCMD" in
            "filter")
                python3 mark_images.py filter -d images -l filterData.txt -o filterData.txt
            ;;
            "coarse")
                python3 mark_images.py coarse -d images -l data.txt -o data.txt
                NUM_TRAINING="$(grep '^images' data.txt | wc | awk '{print int($1*.75)}')"
                NUM_LINES="$(cat data.txt | \
                    awk "/^images/ {c++} c==$NUM_TRAINING {print NR; c++} END {print 0}")"
                cat data.txt | awk "NR <  $NUM_LINES" > trainingDataCoarse.txt
                cat data.txt | awk "NR >= $NUM_LINES" > testingDataCoarse.txt
                rm data.txt
            ;;
            "detailed")
                python3 mark_images.py detailed -d images -l data.txt -o data.txt
                NUM_TRAINING="$(grep '^images' data.txt | wc | awk '{print int($1*.75)}')"
                NUM_LINES="$(cat data.txt | awk "/^images/ {c++} c==$NUM_TRAINING {print NR; c++}")"
                cat data.txt | awk "NR <  $NUM_LINES" > trainingData.txt
                cat data.txt | awk "NR >= $NUM_LINES" > testingData.txt
                rm data.txt
            ;;
            *)
                echo "$USAGE"
                exit 1
            ;;
        esac
    ;;
    "view")
        case "$SUBCMD" in
            "filter")
                python3 mark_images.py filter -d images -l filterData.txt >/dev/null
            ;;
            "coarse")
                python3 mark_images.py coarse -d images -l trainingDataCoarse.txt >/dev/null
            ;;
            "detailed")
                python3 mark_images.py detailed -d images -l trainingData.txt >/dev/null
            ;;
            *)
                echo "$USAGE"
                exit 1
            ;;
        esac
    ;;
    "train")
        case "$SUBCMD" in
            "coarse")
                if [[ trainingDataCoarse.txt -nt dataCoarse_train.npz ]]; then
                    python3 use_network.py train trainingDataCoarse.txt testingDataCoarse.txt \
                        filterData.txt -c -o dataCoarse -s 100 "$@"
                else
                    python3 use_network.py train dataCoarse_train.npz dataCoarse_test.npz \
                        filterData.txt -c -s 100 "$@"
                fi
            ;;
            "detailed")
                if [[ trainingData.txt -nt dataDetailed_train.npz ]]; then
                    python3 use_network.py train trainingData.txt testingData.txt \
                        filterData.txt -o dataDetailed -s 100 "$@"
                else
                    python3 use_network.py train dataDetailed_train.npz dataDetailed_test.npz \
                        filterData.txt -s 100 "$@"
                fi
            ;;
            *)
                echo "$USAGE"
                exit 1
            ;;
        esac
    ;;
    "test")
        case "$SUBCMD" in
            "coarse")
                if [[ testingDataCoarse.txt -nt dataCoarse_test.npz ]]; then
                    python3 use_network.py test testingDataCoarse.txt \
                        filterData.txt -c -o dataCoarse_test -s 100 "$@"
                else
                    python3 use_network.py test dataCoarse_test.npz \
                        filterData.txt -c -s 100 "$@"
                fi
            ;;
            "detailed")
                if [[ testingData.txt -nt dataDetailed_test.npz ]]; then
                    python3 use_network.py test testingData.txt \
                        filterData.txt -o dataDetailed_test -s 100 "$@"
                else
                    python3 use_network.py test dataDetailed_test.npz \
                        filterData.txt -s 100 "$@"
                fi
            ;;
            *)
                echo "$USAGE"
                exit 1
            ;;
        esac
    ;;
    "run")
        RANDOM_IMAGE="$(ls images | sort -R | head -n 1)"
        case "$SUBCMD" in
            "coarse")
                python3 use_network.py run images/"$RANDOM_IMAGE" filterData.txt -c -o out.jpg
            ;;
            "detailed")
                python3 use_network.py run images/"$RANDOM_IMAGE" filterData.txt -o out.jpg
            ;;
            *)
                echo "$USAGE"
                exit 1
            ;;
        esac
    ;;
    "samples")
        case "$SUBCMD" in
            "coarse")
                if [[ trainingDataCoarse.txt -nt dataCoarse_train.npz ]]; then
                    python3 use_network.py samples trainingDataCoarse.txt filterData.txt -c -o out.jpg
                else
                    python3 use_network.py samples dataCoarse_train.npz filterData.txt -c -o out.jpg
                fi
                
            ;;
            "detailed")
                if [[ trainingData.txt -nt dataDetailed_train.npz ]]; then
                    python3 use_network.py samples trainingData.txt filterData.txt -o out.jpg
                else
                    python3 use_network.py samples dataDetailed_train.npz filterData.txt -o out.jpg
                fi
            ;;
            *)
                echo '$USAGE'
                exit 1
            ;;
        esac
    ;;
    *)
        echo "$USAGE"
        exit 1
    ;;
esac

