export CUDA_VISIBLE_DEVICES=0,1

PATH_ONE=/data/.cache/modelscope/models/Qwen/Qwen2.5-VL-7B-Instruct
PATH_TWO=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct


python -m vllm.entrypoints.openai.api_server \
    --model $PATH_TWO \
    --host 0.0.0.0 \
    --port 8080 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --trust_remote_code \
    --allow-credentials \
    --allowed-local-media-path /root/LMUData\
    --limit-mm-per-prompt image=10 \
    --api-key sk-proj-1234567890 \
    --served-model-name Qwen2.5-VL-7B-Instruct-eval