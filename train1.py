from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback

# from new import CarEnv
import new  
import os
import time
import re

def extract_step_count(filename):
    match = re.search(r'rl_model_(\d+)_steps.zip', filename)
    if match:
        return int(match.group(1))
    return None

class MyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MyCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        print("Training is starting!")

    def _on_step(self) -> bool:
        print("Iteration completed!")
        return True 

    def _on_training_end(self) -> None:
        print("Training is complete!")

callback = MyCallback()
checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path="./model_checkpoints/")


print('This is the start of the training script')

print('setting folders for logs and models')
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

print('connecting to env..')

env = new.CarEnv()

env.reset()
print('Env has been reset as part of launch')
if os.path.exists('./model_checkpoints/'):
    latest_checkpoint = max(filter(None, (extract_step_count(file) for file in os.listdir('./model_checkpoints/') if file.endswith('.zip'))))
    model = DQN.load(f'./model_checkpoints/rl_model_{latest_checkpoint}_steps.zip', env=env, tensorboard_log=logdir)
    print(f"Loaded model from checkpoint at iteration {latest_checkpoint}")
else:
    model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.001, tensorboard_log=logdir)

TIMESTEPS = 500_000  
iters = 0
try:
    while iters < 2: 
        iters += 1
        print('Iteration ', iters, ' is to commence...')
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DQN", callback= checkpoint_callback)
        print('Iteration ', iters, ' has been trained')
        model.save(f"{models_dir}/{TIMESTEPS * iters}")
except KeyboardInterrupt:
    # If user interrupts (Ctrl+C), join the thread and exit
    new.stop_event.set()
    new.thread.join(1)
    print("Exiting program...")
    exit(0)