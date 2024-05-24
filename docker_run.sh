#!/bin/bash

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line

docker run -it \
  --user=root \
  --runtime=nvidia \
  --rm \
  --network host \
  --ipc=host \
  --mount src=$(pwd),target=/home/mambauser/code/rl_zoo3,type=bind \
  pmolodo-huggingface-drl:latest \
  bash -c "cd /home/mambauser/code/rl_zoo3/ && $cmd_line"
