# OutRL

"Outside-the-Box Reinforcement Learning"

OutRL is a flexible-but-minimal reinforcement learning library.
Unlike most RL libraries, OutRL is not focused on existing benchmarks, but
on providing a flexible tool for performing RL in as many circumstances as possible.

It provides a single learning algorithm optimized for reliability and
performance when training large models.

OutRL avoids defining any environment API at all, and only defines a minimal
[Trainer and agent API](src/outrl/rl.py). This avoids the use of any "Space"
types, such as the "gym.spaces.Box" that pervades other RL libraries. For
working with the popular gym(nasium) API, a small [optional
library](src/outrl/gym_utils.py) and [example](examples/gym_example.py) is
available.

This makes it easy to train, RNNs, transformers, LLMs, multi-agent RL,
variable-timestep RL, or complex action spaces that combine discrete and
continuous actions.


## The RL Algorithm

OutRL defines a bespoke RL algorithm.
This algorithm is suitable for on-policy, off-policy, and off-line training, or any combination thereof. This is achieved by combining off-policy advantage estimates computed using V-Trace with an AWR style log-likelihood loss and regularization similar to that used in V-MPO.

At each training step this RL algorithm:
  1. Pre-trains a value function to predict future returns from frozen state encodings.
  2. Estimates advantages using the trained value function.
  3. Optimizes the agent to increase the likelihood of positive advantages and improve the encodings' usefulness for predicting future returns.
  4. Performs additional value function training on the frozen state encodings.

By performing most of the value function updates on frozen encodings,
OutRL significantly reduces the number of full forward passes required to
propagate rewards back through time.


## API

OutRL does not require you to use any particular command line user interface, you can use whatever control flow you'd like to create and invoke a `outrl.Trainer` on your agent.

However, there are some utilities for writing short "launch scripts" with a consistent command line interface.

### `Trainer` API

The `Trainer` is the class that provides most of OutRL's functionality.

A Trainer is constructed from a `TrainerConfig` (referred to as `cfg` elsewhere) and an agent.

The agent should be a `torch.nn.Module` with a forward method that takes in a list of "episodes" and returns a list of `ActorOutput` (one per episode).
The agent should also have an integer field `state_encoding_size` that is the
dimensionality of the state encodings returned by the agent.

The episode can be any value you would like, as long as your agent can produce differentiable state encodings and action log-likelihoods for every time-step in the episode.

Besides training, the `Trainer` also implements checkpoint / resume.

Methods:
- `Trainer.add_episode`: Add an episode to the replay buffer. Must be called
  before `train_step`.
- `Trainer.train_step`: Run a training step on the agent.
- `Trainer.add_eval_stats`: Add a dictionary of training statistics. Used for
  checkpointing the "best" agent. Also used in hyper-parameter tuning.

- `Trainer.attempt_resume`: Attempts to resume from the run directory
  (`cfg.log_dir`/`cfg.run_name`).
- `Trainer.maybe_checkpoint`: Checkpoint the `state_dict` to the run directory
  depending on `cfg.checkpoint_best` and the checkpoint interval specified in
  `cfg.checkpoint_interval`.

- `Trainer.state_dict`: Compute a state dictionary for the Trainer (as in
  `torch.nn.Module`).
- `Trainer.load_state_dict`: Load a state dictionary for the Trainer (as in
  `torch.nn.Module`).

### `ExperimentInvocation` API

There's a small utility for managing config files, setting up run directories,
command line parsing, and hyper-parameter tuning.

To use it, write a training function that should receive a (subclass of) `outrl.TrainerConfig`, and will train a new agent given that config.
Then, pass that function and the config type to `outrl.ExperimentInvocation` and call `run()`.

Example:

```python3
# my_launcher.py

class MyConfig(outrl.TrainerConfig):
  env_name: str

def train(cfg: MyConfig):
  env = gym.make(cfg.env_name)
  agent = outrl.gym_utils.make_gym_agent(env, ...)
  trainer = outrl.Trainer(MyConfig, agent)
  ...

if __name__ == '__main__':
  outrl.ExperimentInvocation(train, MyConfig).run()
```

Then, you can call e.g. `python my_launcher.py train --env_name=CartPole-v1`.

If you'd like to add additional options to the command line parser (outside of
the config), you can do so before calling `run()`.

You can tune hyper parameters using the `tune` command:
`python my_launcher.py tune`.

### `outrl.gym_utils`

Because the OpenAI Gym / Farama Gymnasium API is used by so many environments,
some optional tools for working with it are available in `outrl.gym_utils`.

Most of the API is used in [`gym_example.py`](examples/gym_example.py).