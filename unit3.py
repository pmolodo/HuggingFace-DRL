#!/usr/bin/env python3

from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

env_name = "SpaceInvadersNoFrameskip-v4"
config_txt = f'''\
{env_name}:
  env_wrapper:
    - stable_baselines3.common.atari_wrappers.AtariWrapper
  frame_stack: 4
  policy: 'CnnPolicy'
  n_timesteps: !!float 1e6
  buffer_size: 100000
  learning_rate: !!float 1e-4
  batch_size: 32
  learning_starts: 100000
  target_update_interval: 1000
  train_freq: 4
  gradient_steps: 1
  exploration_fraction: 0.1
  exploration_final_eps: 0.01
  # If True, you need to deactivate handle_timeout_termination
  # in the replay_buffer_kwargs
  optimize_memory_usage: False
'''

config_path = "./dqn.yml"
with open(config_path, "w", encoding="utf8") as writer:
    writer.write(config_txt)

increment_timesteps = 200_000
eval_timesteps = 5000

import os
from typing import List, Optional, Tuple

HFUSER = "pmolodo"
env_prefix = env_name + "_"

def get_log_dirs() -> List[Tuple[int, str]]:
    if not os.path.isdir("logs/dqn"):
        return []
    log_dirs = [entry.name for entry in os.scandir("logs/dqn") if entry.is_dir() and entry.name.startswith(env_prefix)]
    log_nums = [int(x[len(env_prefix):]) for x in log_dirs]
    return sorted(zip(log_nums, log_dirs))

def get_last_log_dir() -> Optional[Tuple[int, str]]:
    logs = get_log_dirs()
    if logs:
        return logs[-1][1]

def get_last_log_num() -> int:
    logs = get_log_dirs()
    if logs:
        return logs[-1][0]
    return 0

import subprocess

def sh(cmd: str, **kwargs):
    return subprocess.run(cmd, shell=True, **kwargs)

sh('''huggingface-cli login''')

# Resume training from hugging face hub
#next_log_num = get_last_log_num() + 1
#next_log_dir = env_prefix + str(next_log_num)
#print(f"{next_log_dir=}")

# Download model and save it into the logs/ folder
sh(f'''python -m rl_zoo3.load_from_hub --algo dqn --env "{env_name}" -orga "{HFUSER}" -f logs/''')
last_log_dir = get_last_log_dir()
print(f"{last_log_dir=}")

sh(f'''python -m rl_zoo3.train -i "logs/dqn/{last_log_dir}/{env_name}.zip" --algo dqn --env "{env_name}"  -f logs/  -c ./dqn.yml -n {increment_timesteps}''')
sh(f'''python -m rl_zoo3.enjoy  --algo dqn  --env "{env_name}"  --no-render  --n-timesteps {eval_timesteps} --folder logs/''')
sh(f'''python -m rl_zoo3.push_to_hub --algo dqn  --env {env_name}  --repo-name "dqn-{env_name}"  -orga "{HFUSER}"  -f logs/''')