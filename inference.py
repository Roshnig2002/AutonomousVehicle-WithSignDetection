from stable_baselines3 import DQN
import new
import time

env = new.CarEnv()

print('Connecting to env...')
env.reset()
print('Env has been reset as part of launch') 
# model = DQN.load(r"E:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\model_checkpoints\rl_model_1900000_steps.zip", env=env)
model = DQN.load(r"E:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\model_checkpoints\rl_model_2390000_steps.zip", env=env)
# Inference loop
# Define the maximum number of timesteps per episode
MAX_TIMESTEPS_PER_EPISODE = 500000  # Adjust this value as needed

# In the inference loop
total_rewards = []
num_episodes = 20
try:

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        timestep = 0  # Initialize timestep counter
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            reward_str = str(reward)
            # done_str = str(done)
            text_data = "Class: " + ",".join(map(str, obs[1])) + "\t"
            text_data += "Velocity: " + ",".join(map(str, obs[0])) + "\t"
            text_data += "BrakeValue: " + ",".join(map(str, obs[2])) + "\t"
            text_data += f"Reward: {reward_str}\n"
            # text_data += f"Done: {done_str}\n"

            file_path = "inference.txt"
            with open(file_path,'a') as file:
                file.write(text_data)
            episode_reward += reward
            timestep += 1
            if timestep >= MAX_TIMESTEPS_PER_EPISODE:
                done = True  # End episode if maximum timesteps reached
        print(f"Episode {episode + 1} reward: {episode_reward}")
        total_rewards.append(episode_reward)


except KeyboardInterrupt:
    # If user interrupts (Ctrl+C), join the thread and exit
    stop_event.set()
    thread.join()
    print("Exiting program...")

# Calculate average reward over all episodes
average_reward = sum(total_rewards) / num_episodes
print(f"Average reward over {num_episodes} episodes: {average_reward}")

env.close()  # Close the environment after inference
 