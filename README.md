# `Model_Based_Reinforcement_Learning_JAX`
This is a working version of Kurtland Chua's [repo](https://github.com/kchua/mbrl-jax.git). The repo has been modified to work with newer versions `jax` and `gym`, since JAX has [officially stopped supporting `CUDA 11`](https://jax.readthedocs.io/en/latest/changelog.html#jaxlib-0-4-26-april-3-2024). This version works on `CUDA 12`.
Currently only implements [PETS](https://arxiv.org/abs/1805.12114), MBPO and a Model-based Policy Agent.
There aren't many usable implementations of these MBRL algorithms in JAX, which makes this a valuable resource for runnning MBRL experiments using JAX.

**Warning**: This is a work-in-progress, and has not been evaluated on harder environments!

## Installing Dependencies

A `Dockerfile` with all required dependencies is provided in the `/docker/` folder. This is different from the original Dockerfile provided by Kurt, and uses `Python 3.10.12` along with `jax-v0.4.16`, `gym-v0.26.2` and `mujoco_py-v2.1.2.14`. There are some Cython compilation issues with mujoco_py in `singularity` environments running on a `SLURM` system: [#687](https://github.com/openai/mujoco-py/issues/687), [#644](https://github.com/openai/mujoco-py/issues/644). The Dockerfile in this repo uses a [modified version of `mujoco_py`](https://github.com/avirupdas55/mujoco-py) which is compatible with Singularity, run the `docker_build.sh` with appropriate tags. Alternatively use a prebuilt container: `docker pull avirupdas55/jax:kchua`.

## Running Experiments

A starter script for running an example experiment on cartpole is provided in `model_based_experiment.py`.
This script can be run via

```
  python3 model_based_experiment.py
      --logdir                   DIR      (optional)    Directory for saving checkpoints and 
                                                        rollout recordings. 
      --save-every               FREQ     (optional)    Saving frequency. Defaults to 1 (i.e. 
                                                        save after every iteration)
      --keep-all-checkpoints              (optional)    Flag which enables saving of all 
                                                        checkpoints (instead of only the most 
                                                        recent one).
      --iters                    ITERS    (optional)    Number of training iterations to run.
                                                        Defaults to 100.
      -s                         SEED     (optional)    Experiment random seed. If not 
                                                        provided, uniformly chosen in 
                                                        [0, 10000).
      env                        ENV      (required)    Experiment environment. Currently 
                                                        supports [`MujocoCartpole-v0`,
                                                        `HalfCheetah-v3`]
      agent_type                 AGENT    (required)    Agent type. Choices: [PETS, Policy].
```

For example, to run PETS and save recordings of rollouts to `/external/`:

```
python3 model_based_experiment.py --logdir /external/ MujocoCartpole-v0 PETS
```
