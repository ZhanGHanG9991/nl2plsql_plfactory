set -x
export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_API_KEY='975b4b9d4759ebc4c5ee764c39fe0e9034ae64f5'

# 定义源模型路径和转换后的目标路径
ORIGINAL_MODEL_PATH="/workspace/opt/models/plfactory-3b-plforge-sft-v6-cp3"
FIXED_MODEL_PATH="/workspace/opt/models/plfactory-3b-plforge-sft-v6-cp3-bf16"

# -----------------------------------------------------------------------------
# 步骤 2: 启动训练 (使用修复后的模型路径 FIXED_MODEL_PATH)
# -----------------------------------------------------------------------------
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/opt/projects/verlpl/examples/results/plfactory_rl_train_new_2.parquet \
    data.val_files=/workspace/opt/projects/verlpl/examples/results/plfactory_rl_test_new_2.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${FIXED_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rl-plfactory' \
    trainer.experiment_name='plfactory-plforge-3b-rl-train-4' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 $@