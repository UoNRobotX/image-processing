#!/bin/bash

set -e

USAGE="Usage:
    $0 generate
    $0 mark filter|coarse|detailed [keep]
    $0 view filter|coarse|detailed
    $0 train coarse|detailed
    $0 test coarse|detailed
    $0 run coarse|detailed
    $0 samples coarse|detailed
"

CMD="$1"
TYPE="$2"
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
        case "$TYPE" in
            "filter")
                python3 mark_images.py filter -d images -l data_filter.txt -o data_filter.txt
            ;;
            "coarse")
                python3 mark_images.py coarse -d images -l data.txt -o data.txt
                if [[ "$1" != "keep" ]]; then
                    NUM_TRAINING="$(grep '^images' data.txt | wc | awk '{print int($1*.75)}')"
                    NUM_LINES="$(cat data.txt | \
                        awk "/^images/ {c++} c==$NUM_TRAINING {print NR; c++} END {print 0}")"
                    cat data.txt | awk "NR <  $NUM_LINES" > data_coarse_train.txt
                    cat data.txt | awk "NR >= $NUM_LINES" > data_coarse_test.txt
                    rm data.txt
                fi
            ;;
            "detailed")
                python3 mark_images.py detailed -d images -l data.txt -o data.txt
                if [[ "$1" != "keep" ]]; then
                    NUM_TRAINING="$(grep '^images' data.txt | wc | awk '{print int($1*.75)}')"
                    NUM_LINES="$(cat data.txt | awk "/^images/ {c++} c==$NUM_TRAINING {print NR; c++}")"
                    cat data.txt | awk "NR <  $NUM_LINES" > data_detailed_train.txt
                    cat data.txt | awk "NR >= $NUM_LINES" > data_detailed_test.txt
                    rm data.txt
                fi
            ;;
            *)
                echo "$USAGE"
                exit 1
            ;;
        esac
    ;;
    "view")
        case "$TYPE" in
            "filter")
                python3 mark_images.py filter -d images -l data_filter.txt >/dev/null
            ;;
            "coarse")
                python3 mark_images.py coarse -d images -l data_coarse_train.txt >/dev/null
            ;;
            "detailed")
                python3 mark_images.py detailed -d images -l data_detailed_train.txt >/dev/null
            ;;
            *)
                echo "$USAGE"
                exit 1
            ;;
        esac
    ;;
    "train")
        case "$TYPE" in
            "coarse")
                if [[ data_coarse_train.txt -nt data_coarse_train.npz ]]; then
                    python3 use_network.py train data_coarse_train.txt data_coarse_test.txt \
                        data_filter.txt -c -o data_coarse -s 100 "$@"
                else
                    python3 use_network.py train data_coarse_train.npz data_coarse_test.npz \
                        data_filter.txt -c -s 100 "$@"
                fi
            ;;
            "detailed")
                if [[ data_detailed_train.txt -nt data_detailed_train.npz ]]; then
                    python3 use_network.py train data_detailed_train.txt data_detailed_test.txt \
                        data_filter.txt -o data_detailed -s 100 "$@"
                else
                    python3 use_network.py train data_detailed_train.npz data_detailed_test.npz \
                        data_filter.txt -s 100 "$@"
                fi
            ;;
            *)
                echo "$USAGE"
                exit 1
            ;;
        esac
    ;;
    "test")
        case "$TYPE" in
            "coarse")
                if [[ data_coarse_test.txt -nt data_coarse_test.npz ]]; then
                    python3 use_network.py test data_coarse_test.txt \
                        data_filter.txt -c -o data_coarse_test -s 100 "$@"
                else
                    python3 use_network.py test data_coarse_test.npz \
                        data_filter.txt -c -s 100 "$@"
                fi
            ;;
            "detailed")
                if [[ data_detailed_test.txt -nt data_detailed_test.npz ]]; then
                    python3 use_network.py test data_detailed_test.txt \
                        data_filter.txt -o data_detailed_test -s 100 "$@"
                else
                    python3 use_network.py test data_detailed_test.npz \
                        data_filter.txt -s 100 "$@"
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
        case "$TYPE" in
            "coarse")
                python3 use_network.py run images/"$RANDOM_IMAGE" data_filter.txt -c -o out.jpg "$@"
            ;;
            "detailed")
                python3 use_network.py run images/"$RANDOM_IMAGE" data_filter.txt -o out.jpg "$@"
            ;;
            *)
                echo "$USAGE"
                exit 1
            ;;
        esac
    ;;
    "samples")
        case "$TYPE" in
            "coarse")
                if [[ data_coarse_train.txt -nt data_coarse_train.npz ]]; then
                    python3 use_network.py samples data_coarse_train.txt data_filter.txt \
                        -c -o out.jpg "$@"
                else
                    python3 use_network.py samples data_coarse_train.npz data_filter.txt \
                        -c -o out.jpg "$@"
                fi
                
            ;;
            "detailed")
                if [[ data_detailed_train.txt -nt data_detailed_train.npz ]]; then
                    python3 use_network.py samples data_detailed_train.txt data_filter.txt \
                        -o out.jpg "$@"
                else
                    python3 use_network.py samples data_detailed_train.npz data_filter.txt \
                        -o out.jpg "$@"
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

