set -x
ENGINE=${1:-vllm}
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS



## edit the dataset_dir
data_dir=path/to/dataset/SAT

train_files="['$data_dir/train-00000-of-00004.parquet', '$data_dir/train-00001-of-00004.parquet', '$data_dir/train-00002-of-00004.parquet', '$data_dir/train-00003-of-00004.parquet']"

SYSTEM_PROMPT="|
  You are Actial, a helpful assistant with excellent reasoning ability.
  A user asks you a question, and you should try to solve it.
  You should first think about the reasoning process in the mind and then provides the user with the answer.
  The reasoning process and answer are enclosed within <think> </think> and
  <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
  <answer> answer here </answer>"

contents="['model', 'optimizer', 'extra', 'hf_model']"


## edit the model path
model_path=/path/to/model/checkpoint

## we use the reward function in the reward_score directory
# reward_function_path=verl/utils/reward_score/reward_for_2.py
# reward_function_name=compute_score



python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files=$data_dir/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=16000 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.image_key=images \
    data.system_prompt="'$SYSTEM_PROMPT'" \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.checkpoint.contents="$contents" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.limit_images=100 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=verl/utils/reward_score/reward_custom.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_sat' \
    trainer.experiment_name='qwen2_5_vl_7b_mix_500step' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 $@
