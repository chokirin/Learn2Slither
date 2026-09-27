"""Neural-network agent and model loading helpers for the Slither game."""

import random
from typing import List, Dict, Any
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from board import State, Direction, EMPTY, GREEN

N_ACTIONS = 4
INPUT_SIZE = 21   # State.to_vector(): 20 vision floats plus direction.


# ══════════════════════════════════════════════════════════════════════
#  Base Agent
# ══════════════════════════════════════════════════════════════════════

class BaseAgent:
    def __init__(self,
                 alpha: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.99995,
                 learning: bool = True):
        """Initialize the agent with the given parameters."""
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning = learning
        # current episode number (for logging / saving)
        self.episode = 0
        self.total_steps = 0

    # ------------------------------------------------------------------
    # Action selection — ε-greedy
    # ------------------------------------------------------------------
    def select_action(self, state: State) -> int:
        """
        With probability ε pick a random action (explore),
        otherwise pick the action with the highest Q-value (exploit).
        """
        green_action = self._green_action(state)
        if green_action is not None:
            return green_action
        if self.learning and random.random() < self.epsilon:
            safe_actions = self._safe_actions(state)
            return random.choice(safe_actions or list(range(N_ACTIONS)))
        return self._greedy_action(state)

    def _green_action(self, state: State):
        """Return the safe turn toward the nearest unobstructed green apple."""
        candidates = []
        for action in self._safe_actions(state):
            ray = state.vision[Direction(action)]
            for distance, cell in enumerate(ray, start=1):
                if cell == GREEN:
                    candidates.append((distance, action))
                    break
                if cell != EMPTY:
                    break
        return min(candidates)[1] if candidates else None

    def _safe_actions(self, state: State) -> List[int]:
        """Prefer actions that avoid collisions and red apples immediately."""
        safe = []
        safe_without_red = []
        for action in range(N_ACTIONS):
            if action == (int(state.direction) + 2) % N_ACTIONS:
                continue
            ray = state.vision[Direction(action)]
            if ray and ray[0] not in (1, 3):
                safe.append(action)
                if ray[0] != 5:
                    safe_without_red.append(action)
        return safe_without_red or safe

    def _greedy_action(self, state: State) -> int:
        """Return argmax Q(state, ·). Implemented by subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Q update — implemented by subclasses
    # ------------------------------------------------------------------
    def update(self, state: State, action: int,
               reward: float, next_state: State, done: bool):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------
    def on_episode_end(self):
        """Call once per episode after the terminal step."""
        self.episode += 1
        if self.learning:
            # Decay exploration rate toward epsilon_min
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Save / load — implemented by subclasses
    # ------------------------------------------------------------------
    def save(self, path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, learning: bool = True) -> "BaseAgent":
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Stats dict — used by Trainer and View
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "episode":     self.episode,
            "epsilon":     round(self.epsilon, 4),
            "total_steps": self.total_steps,
            "learning":    self.learning,
        }

# ══════════════════════════════════════════════════════════════════════
#  Neural Network — feed-forward DQN backbone (PyTorch)
# ══════════════════════════════════════════════════════════════════════


class DQN(nn.Module):
    """
    Simple 3-layer MLP.
    Input  : 20 floats (State.to_vector())
    Output : 4 Q-values (one per action)
    """

    def __init__(self, input_size: int = INPUT_SIZE,
                 hidden: int = 128, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════
#  NN Agent — DQN with experience replay + target network
# ══════════════════════════════════════════════════════════════════════
class NNAgent(BaseAgent):
    """
    Deep Q-Network (Mnih et al. 2015).
    Two stabilisation tricks on top of plain Q-learning:
      1. Experience replay  — break correlation between consecutive steps
    2. Target network     — frozen copy of weights, preventing moving targets
    """

    BATCH_SIZE = 64
    MEMORY_SIZE = 10_000
    TARGET_UPDATE_FREQ = 500     # steps between target network syncs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Policy network — trained every step
        self.policy_net = DQN().to(self.device)

        # Target network — frozen, synced every TARGET_UPDATE_FREQ steps
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        self.memory: deque = deque(maxlen=self.MEMORY_SIZE)
        self._steps_since_target = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def _q_values(self, state: State) -> torch.Tensor:
        x = torch.tensor(state.to_vector(),
                         dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(x).squeeze(0)

    def _greedy_action(self, state: State) -> int:
        q_values = self._q_values(state).clone()
        preferred = self._safe_actions(state)
        if preferred:
            for action in range(N_ACTIONS):
                if action not in preferred:
                    q_values[action] = float("-inf")
        return int(q_values.argmax())

    def _action_values(self, state: State):
        return self._q_values(state).tolist()

    # ------------------------------------------------------------------
    # Experience replay
    # ------------------------------------------------------------------
    def _remember(self, state: State, action: int, reward: float,
                  next_state: State, done: bool):
        """Store a transition in the replay buffer."""
        self.memory.append((
            state.to_vector(),
            action,
            reward,
            next_state.to_vector(),
            done,
        ))

    def _train_batch(self):
        """Sample a random mini-batch and perform one gradient step."""
        if len(self.memory) < self.BATCH_SIZE:
            return

        batch = random.sample(self.memory, self.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        S = torch.tensor(states,      dtype=torch.float32).to(
            self.device)  # (B, 21)
        NS = torch.tensor(next_states, dtype=torch.float32).to(
            self.device)  # (B, 21)
        A = torch.tensor(actions,     dtype=torch.long).to(
            self.device)     # (B,)
        R = torch.tensor(rewards,     dtype=torch.float32).to(
            self.device)  # (B,)
        D = torch.tensor(dones,       dtype=torch.float32).to(
            self.device)  # (B,)

        # Q(s, a) — only the action that was actually taken
        q_pred = self.policy_net(S).gather(
            1, A.unsqueeze(1)).squeeze(1)     # (B,)

        # Bellman target using the frozen target network
        with torch.no_grad():
            q_next = self.target_net(NS).max(
                dim=1).values                   # (B,)
        q_target = R + self.gamma * q_next * \
            (1.0 - D)                      # (B,)

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ------------------------------------------------------------------
    # Update — called every step by Trainer
    # ------------------------------------------------------------------
    def update(self, state: State, action: int,
               reward: float, next_state: State, done: bool):
        if not self.learning:
            return
        self.total_steps += 1

        self._remember(state, action, reward, next_state, done)
        self._train_batch()

        # Sync target network periodically
        self._steps_since_target += 1
        if self._steps_since_target >= self.TARGET_UPDATE_FREQ:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self._steps_since_target = 0

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            "type": "neural",
            "hyperparams": {
                "alpha":         self.alpha,
                "gamma":         self.gamma,
                "epsilon":       self.epsilon,
                "epsilon_min":   self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            "stats": {
                "episode":     self.episode,
                "total_steps": self.total_steps,
            },
            "weights": self.policy_net.state_dict(),
        }, path)
        print(f"[NN] Saved → {path}")

    @classmethod
    def load(cls, path: str, learning: bool = True) -> "NNAgent":
        data = torch.load(path, map_location="cpu")
        assert data["type"] == "neural", (
            f"Expected neural model, got {data['type']}"
        )
        hp = data["hyperparams"]
        agent = cls(learning=learning, **hp)
        agent.policy_net.load_state_dict(data["weights"])
        agent.target_net.load_state_dict(data["weights"])
        agent.episode = data["stats"]["episode"]
        agent.total_steps = data["stats"]["total_steps"]
        print(f"[NN] Loaded ← {path}  (ep {agent.episode})")
        return agent

    def stats(self) -> Dict[str, Any]:
        s = super().stats()
        s["type"] = "neural"
        s["memory"] = len(self.memory)
        s["device"] = str(self.device)
        return s


# ══════════════════════════════════════════════════════════════════════
#  Factory functions
# ══════════════════════════════════════════════════════════════════════
def make_agent(model_type: str = "neural",
               learning: bool = True, **kwargs) -> BaseAgent:
    """Instantiate the neural agent."""
    if model_type == "neural":
        return NNAgent(learning=learning, **kwargs)
    raise ValueError(
        f"Unknown model type: '{model_type}'. Only 'neural' is supported.")


def load_agent(path: str, learning: bool = True) -> BaseAgent:
    """Load a neural agent from a saved checkpoint."""
    try:
        meta = torch.load(path, map_location="cpu")
        model_type = meta.get("type")
    except Exception:
        raise ValueError(
            "Invalid neural checkpoint. Legacy checkpoints are no longer "
            "supported."
        )

    if model_type == "neural":
        return NNAgent.load(path, learning=learning)
    raise ValueError(
        f"Unsupported model type in file: '{model_type}'. Only neural "
        "checkpoints are supported."
    )
