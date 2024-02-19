echo "Infer WITHOUT the lora fine-tuning layer"
python ./llava/eval/run_llava.py \
--model-path ./temp/llava-v1.5-7b \
--image-file ./temp/dataset/images/dd874377-cd04-41d8-ba9b-124d03ea68a7.jpg \
--query "why was this photo taken?"
