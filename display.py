"""
view/display.py — Pygame graphical interface for Snake RL
Supports: normal speed, human-readable speed, step-by-step mode, headless mode.
"""

import time

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from board import Board


# ── Colour palette ────────────────────────────────────────────────────────
BG_DARK = (15,  17,  20)
BG_GRID = (25,  28,  35)
CELL_EMPTY = (30,  34,  42)
SNAKE_HEAD = (80, 220, 120)
SNAKE_BODY = (45, 160,  80)
SNAKE_TAIL = (30, 110,  55)
APPLE_G = (60, 200,  60)
APPLE_R = (220,  60,  60)
TEXT_COL = (200, 210, 220)
PANEL_BG = (20,  22,  28)
ACCENT = (80, 220, 120)
DANGER = (220,  70,  70)

SPEEDS = {
    "fast":  0.0,       # no delay (training)
    "normal": 0.10,      # comfortable viewing
    "slow":  0.25,      # human-readable
    "step":  None,      # wait for keypress
}


class Display:
    CELL = 52
    MARGIN = 4
    PANEL_W = 260
    TOP_BAR = 48

    def __init__(self, speed: str = "normal"):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame not installed. Run: pip install pygame")

        pygame.init()
        pygame.display.set_caption("Snake RL")

        board_px = Board.SIZE * self.CELL + (Board.SIZE + 1) * self.MARGIN
        self.width = board_px + self.PANEL_W
        self.height = board_px + self.TOP_BAR

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_sm = pygame.font.SysFont("monospace", 14)
        self.font_xs = pygame.font.SysFont("monospace", 12)

        self.speed = speed
        self.delay = SPEEDS.get(speed, 0.12)
        self.step_mode = (speed == "step")
        self._running = True

    # ── Main render call ───────────────────────────────────────────────────
    def render(self, board: Board, stats: dict):
        """Render one frame. Returns False if window was closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self._running = False
                    return False
                if self.step_mode and event.key == pygame.K_SPACE:
                    return True   # advance one step

        if not self._running:
            return False

        self.screen.fill(BG_DARK)
        self._draw_top_bar(board, stats)
        self._draw_board(board)
        self._draw_panel(board, stats)

        if self.step_mode:
            self._draw_step_hint()

        pygame.display.flip()

        if self.step_mode:
            # Block until SPACE or Q
            return self._wait_for_step()
        else:
            if self.delay:
                time.sleep(self.delay)
            return True

    def _wait_for_step(self) -> bool:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self._running = False
                        return False
                    if event.key in (
                            pygame.K_SPACE, pygame.K_RIGHT, pygame.K_n):
                        return True
            self.clock.tick(30)

    # ── Top bar ────────────────────────────────────────────────────────────
    def _draw_top_bar(self, board: Board, stats: dict):
        pygame.draw.rect(self.screen, PANEL_BG,
                         (0, 0, self.width, self.TOP_BAR))
        title = self.font_lg.render("SNAKE  RL", True, ACCENT)
        self.screen.blit(title, (16, 14))

        ep_txt = self.font_sm.render(
            f"Episode {stats.get('episode', 0)+1}   "
            f"ε={stats.get('epsilon', 0):.3f}   "
            f"Steps {board.steps}   Score {board.score}   "
            f"Len {len(board.snake)}",
            True, TEXT_COL
        )
        self.screen.blit(ep_txt, (160, 16))

    # ── Board grid ────────────────────────────────────────────────────────
    def _cell_rect(self, r, c):
        x = self.MARGIN + c * (self.CELL + self.MARGIN)
        y = self.TOP_BAR + self.MARGIN + r * (self.CELL + self.MARGIN)
        return pygame.Rect(x, y, self.CELL, self.CELL)

    def _draw_board(self, board: Board):
        # Background grid
        total_px = Board.SIZE * self.CELL + (Board.SIZE + 1) * self.MARGIN
        grid_rect = pygame.Rect(0, self.TOP_BAR, total_px, total_px)
        pygame.draw.rect(self.screen, BG_GRID, grid_rect)

        # Cells
        for r in range(Board.SIZE):
            for c in range(Board.SIZE):
                rect = self._cell_rect(r, c)
                pygame.draw.rect(self.screen, CELL_EMPTY,
                                 rect, border_radius=6)

        # Snake body (gradient from body to tail)
        snake = board.snake
        n = len(snake)
        for i, (r, c) in enumerate(reversed(snake[1:])):
            t = i / max(n - 1, 1)
            col = self._lerp_color(SNAKE_TAIL, SNAKE_BODY, t)
            rect = self._cell_rect(r, c)
            pygame.draw.rect(self.screen, col, rect, border_radius=8)

        # Head
        if snake:
            r, c = snake[0]
            rect = self._cell_rect(r, c)
            pygame.draw.rect(self.screen, SNAKE_HEAD, rect, border_radius=10)
            # Eye dot
            eye_x = rect.centerx + 8
            eye_y = rect.centery - 8
            pygame.draw.circle(self.screen, BG_DARK, (eye_x, eye_y), 4)

        # Green apples
        for r, c in board.green_apples:
            rect = self._cell_rect(r, c)
            pygame.draw.ellipse(self.screen, APPLE_G, rect.inflate(-8, -8))

        # Red apple
        if board.red_apple:
            r, c = board.red_apple
            rect = self._cell_rect(r, c)
            pygame.draw.ellipse(self.screen, APPLE_R, rect.inflate(-8, -8))

    @staticmethod
    def _lerp_color(c1, c2, t):
        return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

    # ── Side panel ─────────────────────────────────────────────────────────
    def _draw_panel(self, board: Board, stats: dict):
        board_px = Board.SIZE * self.CELL + (Board.SIZE + 1) * self.MARGIN
        px = board_px + 10
        py = self.TOP_BAR + 10
        lh = 22

        def txt(s, color=TEXT_COL, bold=False):
            nonlocal py
            font = self.font_lg if bold else self.font_sm
            surf = font.render(s, True, color)
            self.screen.blit(surf, (px, py))
            py += lh

        txt("AGENT", ACCENT, bold=True)
        txt(f"Type:    {stats.get('type', '?')}")
        txt(f"Episode: {stats.get('episode', 0)}")
        txt(f"ε (exp): {stats.get('epsilon', 0):.4f}")
        txt(f"Steps:   {stats.get('total_steps', 0)}")
        if "states" in stats:
            txt(f"States:  {stats['states']}")
        if "memory" in stats:
            txt(f"Memory:  {stats['memory']}")
        py += 10

        txt("GAME", ACCENT, bold=True)
        txt(f"Score:   {board.score}")
        txt(f"Length:  {len(board.snake)}")
        txt(f"Steps:   {board.steps}")

        learn_col = ACCENT if stats.get("learning") else DANGER
        learn_str = "ON" if stats.get("learning") else "OFF (eval)"
        py += 10
        txt("LEARNING", ACCENT, bold=True)
        txt(f"Mode:    {learn_str}", learn_col)
        py += 10

        txt("CONTROLS", ACCENT, bold=True)
        txt("Q: quit")
        if self.step_mode:
            txt("SPACE: next step")
        txt(f"Speed: {self.speed}")

    def _draw_step_hint(self):
        hint = self.font_lg.render("STEP MODE — press SPACE", True, ACCENT)
        board_px = Board.SIZE * self.CELL + (Board.SIZE + 1) * self.MARGIN
        x = (board_px - hint.get_width()) // 2
        y = self.TOP_BAR + board_px - 36
        pygame.draw.rect(
            self.screen, PANEL_BG,
            pygame.Rect(x - 10, y - 6, hint.get_width() + 20, 36),
            border_radius=6)
        self.screen.blit(hint, (x, y))

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def show_game_over(self, board: Board, stats: dict):
        """Flash game-over overlay, then pause briefly."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))

        msg = self.font_lg.render(
            f"GAME OVER — Score {board.score}  Len {len(board.snake)}",
            True, DANGER)
        sub = self.font_sm.render("Starting next episode...", True, TEXT_COL)
        cx, cy = self.width // 2, self.height // 2
        self.screen.blit(msg, (cx - msg.get_width() // 2, cy - 20))
        self.screen.blit(sub, (cx - sub.get_width() // 2, cy + 14))
        pygame.display.flip()

        if self.step_mode:
            self._wait_for_step()
        else:
            time.sleep(0.6)

    def close(self):
        pygame.quit()

    @property
    def running(self):
        return self._running


# ── Headless null display ────────────────────────────────────────────────
class HeadlessDisplay:
    """Drop-in replacement when graphical output is disabled."""
    running = True

    def render(self, board, stats):
        return True

    def show_game_over(self, board, stats):
        pass

    def close(self):
        pass


def make_display(headless: bool = False, speed: str = "normal"):
    if headless or not PYGAME_AVAILABLE:
        return HeadlessDisplay()
    return Display(speed=speed)
