#!/usr/bin/env bash

# Function to check if a directory exists and remove it
remove_directory() {
    if [ -d "$1" ]; then
        echo "Removing existing directory: $1"
        rm -rf "$1"
    fi
}

# Remove existing repositories
remove_directory "mujoco-py"

# Clone necessary repositories
git clone -b v2.0.2.14-local-update https://github.com/avirupdas55/mujoco-py.git

DOCKER_BUILDKIT=1 docker build -t jax:kchua . 

docker tag jax:kchua avirupdas55/jax:kchua
#docker login -u avirupdas55
docker push avirupdas55/jax:kchua