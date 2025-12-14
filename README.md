# RobustCan
training environment in ManiSkill/SAPIEN for Deep RL with PPO

This is a project branching out from my previous project, Prompt2Pose.

You need to install ManiSkill library and SAPIEN simulator before you start.

You can use the ppo.py (PPO baseline) from the link below to train a model with my environments.
https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo

These are the parameters I have used to achieve a task success rate of 90%.
--num_envs=1024 --update_epochs=4 --num_minibatches=32 --total-timesteps=80_000_000 --eval_freq=50 --num-steps=200 --num_eval_steps=200 --gamma=0.99 --gae-lambda=0.95 --ent_coef=0.001
