"""Snake RL environment."""

import random
from enum import IntEnum
from typing import List, Optional, Tuple


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


DELTA = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}

EMPTY = 0
WALL = 1
HEAD = 2
BODY = 3
GREEN = 4
RED = 5


class Reward:
    GREEN = 10.0
    RED = -9.0
    STEP = 0.1
    WALL = -20.0
    COLLISION = -20.0
    DEAD = -20.0


class Board:
    SIZE = 10
    N_GREEN = 2
    N_RED = 1
    INIT_LEN = 3
    MAX_STEPS = 500

    def __init__(self):
        self.reset()

    def reset(self) -> "State":
        self.done = False
        self.score = 0
        self.steps = 0

        self._place_snake()
        self.green_apples: List[Tuple[int, int]] = []
        self.red_apple: Optional[Tuple[int, int]] = None
        for _ in range(self.N_GREEN):
            self._spawn_green()
        self._spawn_red()
        return self._get_state()

    def _place_snake(self):
        direction = random.choice(list(Direction))
        dr, dc = DELTA[direction]

        for _ in range(1000):
            head_r = random.randint(0, self.SIZE - 1)
            head_c = random.randint(0, self.SIZE - 1)
            body = [(head_r - i * dr, head_c - i * dc) for i in range(self.INIT_LEN)]
            if all(0 <= row < self.SIZE and 0 <= col < self.SIZE for row, col in body):
                if len(set(body)) == self.INIT_LEN:
                    self.snake = body
                    self.direction = direction
                    return
        raise RuntimeError("Could not place snake after 1000 tries")

    def _free_cells(self) -> List[Tuple[int, int]]:
        occupied = set(self.snake) | set(self.green_apples)
        if self.red_apple:
            occupied.add(self.red_apple)
        return [
            (row, col)
            for row in range(self.SIZE)
            for col in range(self.SIZE)
            if (row, col) not in occupied
        ]

    def _spawn_green(self):
        free = self._free_cells()
        if free:
            self.green_apples.append(random.choice(free))

    def _spawn_red(self):
        free = self._free_cells()
        if free:
            self.red_apple = random.choice(free)

    def step(self, action: int) -> Tuple["State", float, bool]:
        if self.done:
            raise RuntimeError("step() called on finished episode")

        direction = Direction(action)
        if direction == Direction((int(self.direction) + 2) % 4):
            direction = self.direction
        self.direction = direction
        dr, dc = DELTA[direction]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        if not (0 <= new_head[0] < self.SIZE and 0 <= new_head[1] < self.SIZE):
            self.done = True
            return self._get_state(), Reward.WALL, True

        if new_head in self.snake[:-1]:
            self.done = True
            return self._get_state(), Reward.COLLISION, True

        self.snake.insert(0, new_head)
        reward = Reward.STEP

        if new_head in self.green_apples:
            self.green_apples.remove(new_head)
            self.score += 1
            reward = Reward.GREEN
            self._spawn_green()
        elif new_head == self.red_apple:
            self.red_apple = None
            self.score -= 1
            reward = Reward.RED
            self.snake.pop()
            if len(self.snake) > 0:
                self.snake.pop()
            self._spawn_red()
        else:
            self.snake.pop()

        if len(self.snake) == 0:
            self.done = True
            return self._get_state(), Reward.DEAD, True

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.done = True
            return self._get_state(), Reward.STEP, True
        return self._get_state(), reward, False

    def _get_state(self) -> "State":
        if not self.snake:
            vision = {direction: [WALL] for direction in Direction}
            return State(vision, self.direction, 0)

        head = self.snake[0]
        vision = {direction: self._cast_ray(head, direction) for direction in Direction}
        return State(vision, self.direction, len(self.snake))

    def _cast_ray(self, start: Tuple[int, int], direction: Direction) -> List[int]:
        dr, dc = DELTA[direction]
        row, col = start
        ray = []

        while True:
            row += dr
            col += dc
            if not (0 <= row < self.SIZE and 0 <= col < self.SIZE):
                ray.append(WALL)
                break
            if (row, col) == self.snake[0]:
                ray.append(HEAD)
            elif (row, col) in self.snake:
                ray.append(BODY)
            elif (row, col) in self.green_apples:
                ray.append(GREEN)
            elif (row, col) == self.red_apple:
                ray.append(RED)
            else:
                ray.append(EMPTY)
        return ray

    def render_vision(self, state: "State"):
        symbol = {EMPTY: "0", WALL: "W", HEAD: "H", BODY: "S", GREEN: "G", RED: "R"}
        dir_name = {
            Direction.UP: "UP",
            Direction.DOWN: "DOWN",
            Direction.LEFT: "LEFT",
            Direction.RIGHT: "RIGHT",
        }
        print("\n--- Snake Vision ---")
        for direction in Direction:
            ray = state.vision[direction]
            line = " ".join(symbol[cell] for cell in ray)
            print(f"  {dir_name[direction]:5s}: {line}")
        print(f"  Direction: {dir_name[self.direction]}  Length: {len(self.snake)}")

    def render_board(self):
        grid = [["." for _ in range(self.SIZE)] for _ in range(self.SIZE)]
        for row, col in reversed(self.snake):
            grid[row][col] = "S"
        if self.snake:
            head_row, head_col = self.snake[0]
            grid[head_row][head_col] = "H"
        for row, col in self.green_apples:
            grid[row][col] = "G"
        if self.red_apple:
            row, col = self.red_apple
            grid[row][col] = "R"

        border = "+" + "-" * (self.SIZE * 2 - 1) + "+"
        print(border)
        for row in grid:
            print("|" + " ".join(row) + "|")
        print(border)
        print(f"Score: {self.score}  Steps: {self.steps}  Length: {len(self.snake)}")


class State:
    def __init__(self, vision: dict, direction: Direction, length: int):
        self.vision = vision
        self.direction = direction
        self.length = length

    def to_key(self) -> tuple:
        key = []
        for direction in Direction:
            ray = self.vision[direction]
            first = next((cell for cell in ray if cell != EMPTY), ray[-1])
            key.append(first)
        key.append(int(self.direction))
        key.append(self.length)
        return tuple(key)

    def to_vector(self) -> List[float]:
        features = []
        for direction in Direction:
            ray = self.vision[direction]
            total = len(ray)
            dist_wall = total / Board.SIZE
            dist_body = 1.0
            dist_green = 1.0
            dist_red = 1.0

            for index, cell in enumerate(ray):
                norm = (index + 1) / Board.SIZE
                if cell == BODY and dist_body == 1.0:
                    dist_body = norm
                if cell == GREEN and dist_green == 1.0:
                    dist_green = norm
                if cell == RED and dist_red == 1.0:
                    dist_red = norm

            features.extend([dist_wall, dist_body, dist_green, dist_red])

        one_hot = [0.0] * 4
        one_hot[int(self.direction)] = 1.0
        features.extend(one_hot)
        features.append(self.length / (Board.SIZE * Board.SIZE))
        return features