lm_eval --model hf \
    --model_args pretrained=./data/opt-125m \
    --tasks truthfulqa,piqa,ifeval \
    --num_fewshot 3 \
    --output_path ./output \
    --batch_size auto \
    --device cuda:0 \

lm_eval --model hf \
    --model_args pretrained=./data/opt-125m \
    --tasks gsm8k \
    --num_fewshot 8 \
    --output_path ./output \
    --batch_size auto \
    --device cuda:0 \

lm_eval --model hf \
    --model_args pretrained=./data/opt-125m \
    --tasks mmlu \
    --num_fewshot 0 \
    --output_path ./output \
    --batch_size auto \
    --device cuda:0 \
