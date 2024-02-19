# Fine tuning example - from OK-VQA dataset

Ref: https://wandb.ai/byyoung3/ml-news/reports/How-to-Fine-Tune-LLaVA-on-a-Custom-Dataset--Vmlldzo2NjUwNTc1

1. setup LLaVA, with the extra libraries for training

On `Uxix`:

a. Install Packages

```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

b. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

- also see [README](./README.md#install)

2. install git lfs:

```
chmod +x *.sh

./install-lfs__sr.sh
```

3. get/create the data (change to use your data!)

```
pip install datasets

python prep_data__OK-VQA__sr.py
```

4. download the base model:

```
./download_llava_weights__sr.sh
```

5. review the script below, and the comments section.

You need to check the values for xxx to match your hardware.

```
cat ./train_qlora__wandb.sh
```

6. execute the script:

```
./train_qlora__wandb.sh
```

7. monitor the progress - if quality is high enough, you can stop the training early

- if you run out of GPU memory, then adjust the script to offload more work to CPU

8. To test inference with the QLORA layer - run this script:

```
./infer_example.sh
```

To infer with a given prompt and image:

```
./infer_qlora_v1.5__wandb.sh <path to image> "my prompt"
```

To infer WITHOUT the lora layer (to see the behaviour BEFORE fine-tuning):

```
./infer_example__no_lora.sh
```
