export CUDA_VISIBLE_DEVICES=2,3,4,5


# PATH_ONE=/data/.cache/modelscope/models/Qwen/Qwen2.5-VL-7B-Instruct
PATH_TWO=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct


# vllm serve $PATH_TWO \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --data-parallel-size 4 \
#     --max-model-len 32768 \
#     --gpu-memory-utilization 0.9 \
#     --trust_remote_code \
#     --allow-credentials \
#     --limit-mm-per-prompt image=2


python -m vllm.entrypoints.openai.api_server \
    --model $PATH_TWO \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --trust_remote_code \
    --allow-credentials \
    --allowed-local-media-path /cfs-bj/sunhao/datasets/LMUData\
    --limit-mm-per-prompt image=100 \
    --served-model-name Qwen2.5-VL-7B-Instruct