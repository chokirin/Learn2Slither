from collections import deque
import time

class Trainer:
    def __init__(self, agent, board, view, config):
        self.agent  = agent
        self.board  = board
        self.view   = view
        self.cfg    = config        # simple dict or dataclass
        self.recent_scores  = deque(maxlen=100)
        self.recent_lengths = deque(maxlen=100)
        self.best_length    = 0
        self.start_time     = time.time()

    def run(self):
        for episode in range(self.cfg["episodes"]):
            done = self._run_episode()
            if not done:            # window was closed mid-episode
                break
            self._on_episode_end(episode)
        self._finalize()

    def _run_episode(self) -> bool:
        state = self.board.reset()
        while True:
            if not self.view.running:
                return False
            if not self.view.render(self.board, self.agent.stats()):
                return False

            action               = self.agent.select_action(state)
            next_state, reward, done = self.board.step(action)
            self.agent.update(state, action, reward, next_state, done)
            state = next_state

            if done:
                self.view.show_game_over(self.board, self.agent.stats())
                return True

    def _on_episode_end(self, episode: int):
        length = len(self.board.snake)
        self.recent_scores.append(self.board.score)
        self.recent_lengths.append(length)
        if length > self.best_length:
            self.best_length = length
        self.agent.on_episode_end()

        if (episode + 1) % self.cfg["log_every"] == 0:
            self._log(episode)
        if self.cfg["save_every"] and (episode + 1) % self.cfg["save_every"] == 0:
            self.agent.save(self._checkpoint_path(episode + 1))

    def _log(self, episode: int):
        avg_s = sum(self.recent_scores)  / len(self.recent_scores)
        avg_l = sum(self.recent_lengths) / len(self.recent_lengths)
        print(
            f"Ep {episode+1:>6} | ε={self.agent.epsilon:.4f} | "
            f"AvgScore={avg_s:>7.2f} | AvgLen={avg_l:.2f} | "
            f"BestLen={self.best_length} | t={time.time()-self.start_time:.0f}s"
        )

    def _finalize(self):
        if self.agent.learning:
            self.agent.save(self._checkpoint_path(self.cfg["episodes"]))
        self.view.close()

    def _checkpoint_path(self, ep: int) -> str:
        return self.cfg.get("save") or f"models/{self.agent.stats()['type']}_ep{ep}.json"
