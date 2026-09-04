# Snake RL

Reinforcement learning agent for the Snake game using a DQN-style neural agent implemented with **PyTorch**, replay, and a target network.

---

## Project Structure

```
.
├── board.py        # Board, snake, apples, state, rewards
├── agent.py        # BaseAgent, NNAgent, make/load helpers
├── display.py      # Pygame graphical interface + headless mode
├── main.py         # CLI entry point
├── trainer.py      # Training loop and checkpointing
├── models/         # Saved model checkpoints
└── README.md
```
## Bellman Equation

The agent estimates the optimal action-value function with the Bellman optimality equation:

$$
Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]
$$

For each replay transition, the DQN uses this target:

$$
y = r + \gamma (1 - done) \max_{a'} Q_{target}(s', a')
$$

Here, $s$ is the current state, $a$ is the chosen action, $r$ is the reward, $s'$ is the next state, and $\gamma$ is the discount factor. The $(1 - done)$ term prevents the agent from estimating future rewards after the snake has died.

The DQN minimizes the mean squared error(MSE) between the predicted value for the chosen action and the Bellman target:

$$
L(\theta) = \frac{1}{B} \sum_{i=1}^{B} \left(y_i - Q_{policy}(s_i, a_i; \theta)\right)^2
$$

Here, $B$ is the mini-batch size. The target network is kept fixed while this loss is calculated, and the optimizer updates only the policy network.
---

## Rules (from spec)

| Rule | Detail |
|------|--------|
| Board | 10 × 10 cells |
| Apples | 2 green (grow), 1 red (shrink) |
| Snake start | 3 cells, random |
| Wall hit | Game over |
| Self collision | Game over |
| Length → 0 | Game over |

---

## State (Vision)

The snake sees in **4 directions** from its head.  
Each direction casts a ray and records the first entity encountered. The current length is also included in the state.

**NN input vector** — 21 floats:  
For each of 4 directions: `(dist_wall, dist_body, dist_green, dist_red)` + 4-bit one-hot direction + normalized length.

---

## Rewards

| Event | Reward |
|-------|--------|
| Eat green apple | +10 |
| Eat red apple | -9 |
| Normal step | +0.1 |
| Wall / self collision | -20 |
| Length drops to 0 | -20 |

---

## Agents

### NNAgent
PyTorch DQN-style agent:
- Architecture: `21 → 128 → 128 → 4`
- ReLU activations
- Experience replay buffer (10 000 transitions)
- Fixed target network (updated every 500 steps)
- Mini-batch training (batch size 64)
- Adam optimizer with MSE loss
- Uses CUDA automatically when available

---

## Usage

### Train a new model

python main.py --episodes 1000 --model neural
python main.py --episodes 1000 --load models/neural_ep500.json --headless --speed fast

# Neural network, headless (fast), save every 500 eps
python main.py --episodes 2000 --model neural --headless --speed fast --save-every 500

# Neural network, step-by-step mode (press SPACE to advance)
python main.py --episodes 10 --model neural --speed step

# Slow human-readable speed with terminal vision output
python main.py --episodes 20 --model neural --speed slow
```

### Load and continue training

```bash
python main.py --episodes 1000 --load models/neural_ep20.json
```
### Evaluate (no learning)

```bash
python main.py --episodes 100 --load models/neural.json --no-learn --speed normal
python main.py --episodes 100 --load models/neural_ep20.json --no-learn --headless --speed fast
python main.py --episodes 100 --load models/neural_ep1000.json --no-learn --speed normal

When `--load` is provided, the checkpoint must be a neural model. Use `--model neural` when starting from scratch.

### All CLI options

```
--episodes   N        Number of episodes (default: 500)
--model      TYPE     neural (default: neural)
--load       PATH     Load existing model
--save       PATH     Save path (auto-named if omitted)
--save-every N        Checkpoint every N episodes (default: 500)
--speed      SPEED    fast | normal | slow | step (default: normal)
--headless            No graphical display (training mode)
--no-learn            Disable Q-update (evaluation mode)
--log-every  N        Console log every N episodes (default: 50)
--alpha      F        Learning rate (default: 0.001)
--gamma      F        Discount factor (default: 0.95)
--epsilon    F        Initial exploration (default: 1.0)
--eps-decay  F        Epsilon decay per episode (default: 0.995)
--eps-min    F        Minimum epsilon (default: 0.05)
```

### Checkpoints

`trainer.py` saves checkpoints automatically every `--save-every` episodes and again at the end of training.
If you do not pass `--save`, the default name is `models/<type>_ep<N>.json`.

Neural checkpoints are saved with `torch.save()` and contain the model weights plus training metadata, even if the filename ends in `.json`.

---

## Model files

Models are saved through the agent's `save()` method.

- Neural checkpoints are saved as PyTorch checkpoints containing the model weights and training metadata.

In both cases, the default filenames used by `main.py` are written under `models/`.

---

## Controls (graphical mode)

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `SPACE` | Next step (step mode only) |

---

## Dependencies


```bash
source /goinfre/$USER/venv_snake/bin/activate

pip install numpy pygame torch
```
# 1. Point pip cache away from home
export PIP_CACHE_DIR=/goinfre/$USER/pip_cache

# 2. Install pytorch into a virtualenv on goinfre
python3 -m venv /goinfre/$USER/venv_snake

# 3. Activate it
source /goinfre/$USER/venv_snake/bin/activate

# 4. Now install — nothing touches $HOME
pip install torch numpy pygame