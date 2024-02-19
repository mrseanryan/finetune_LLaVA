#!/bin/bash

set -e -x  # stop on 1st error, debug output of args used

function print_usage()
{
    echo "USAGE: $0 <path to image file> <prompt>"
}

PATH_TO_IMAGE=""
PROMPT=""
if [ "$#" == "2" ]; then
    PATH_TO_IMAGE=$1
    PROMPT=$2
else
    print_usage
    exit 42
fi

# To infer with the QLORA layer:
#
python llava/eval/run_llava.py --model-path ./temp/checkpoints/llama-2-7b-chat-task-qlora/best_llava_eval_model \
--model-base ./temp/llava-v1.5-7b \
--image-file $PATH_TO_IMAGE \
--query "$PROMPT"

# example query:
# “why was this photo taken?”

# example path to an image:
# ./temp/dataset/images/0f47c0b5-2c77-45e6-87b0-89af46e99500.jpg
