#!/usr/bin/env python3
import argparse
import os
from board import Board
from agent import make_agent, load_agent
from display import make_display
from trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Snake RL")
    p.add_argument("--episodes",   type=int,   default=500)
    p.add_argument("--model",      type=str,
                   default="neural", choices=["neural"])
    p.add_argument("--load",       type=str,   default=None)
    p.add_argument("--save",       type=str,   default=None)
    p.add_argument("--save-every", type=int,   default=500)
    p.add_argument("--speed",      type=str,   default="normal",
                   choices=["fast", "normal", "slow", "step"])
    p.add_argument("--headless",   action="store_true")
    p.add_argument("--no-learn",   action="store_true")
    p.add_argument("--log-every",  type=int,   default=50)
    p.add_argument("--alpha",      type=float, default=0.001)
    p.add_argument("--gamma",      type=float, default=0.95)
    p.add_argument("--epsilon",    type=float, default=1.0)
    p.add_argument("--eps-decay",  type=float, default=0.995)
    p.add_argument("--eps-min",    type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("models", exist_ok=True)
    learning = not args.no_learn

    agent = (load_agent(args.load, learning=learning) if args.load
             else make_agent(args.model, learning=learning,
                             alpha=args.alpha, gamma=args.gamma,
                             epsilon=args.epsilon,
                             epsilon_decay=args.eps_decay,
                             epsilon_min=args.eps_min))

    config = vars(args)           # flat dict, Trainer reads what it needs
    config["save_every"] = args.save_every

    trainer = Trainer(
        agent=agent,
        board=Board(),
        view=make_display(headless=args.headless, speed=args.speed),
        config=config,
    )
    trainer.run()


if __name__ == "__main__":
    main()
