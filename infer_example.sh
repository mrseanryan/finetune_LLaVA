image_files=(./temp/dataset/images/*)
first_image="${image_files[0]}"

python ./llava/eval/run_llava.py --model-path ./temp/checkpoints/llama-2-7b-chat-task-qlora/best_llava_eval_model \
--model-base ./temp/llava-v1.5-7b \
--image-file $first_image \
--query "why was this photo taken?"
