# agent_train_eval.py
# Visual evolutionary training for Flappy Bird
# - Shows all birds of the current generation in one window
# - Evolves to the next generation without closing the window

import sys
import json
import numpy as np
import pygame

from typing import List, Tuple

# Window / world constants (match your game)
WIN_WIDTH, WIN_HEIGHT = 600, 800
FLOOR_Y = 730
FPS = 30
PIPE_SPAWN_X = 700
PIPE_ADD_GAP = 300  
HUD_FONT = "comicsans"

pygame.init()
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird — Visual Evolution")
font = pygame.font.SysFont(HUD_FONT, 24)

# --- game components (yours) ---
from components.bird import Bird
from components.base import Base
from components.pipes import Pipe
from components.agent import Agent as Policy  # your policy network (forward/act)

       # when last pipe goes left enough, add a new one

# HUD / viz
MAX_VISUAL_BIRDS = 20     # draw at most this many (still simulate all)
BIRD_ALPHA = 200           # transparency so overlaps are visible (0..255)

# ---------------- GA utilities ----------------

class EvoConfig:
    """Evolution hyperparams & network sizes"""
    def __init__(
        self,
        in_dim: int = 4,
        hidden: Tuple[int, ...] = (32, 32),
        out_dim: int = 1,
        population_size: int = 25,
        elite_fraction: float = 0.10,
        mutate_prob: float = 0.12,
        mutate_sigma_w: float = 0.12,
        mutate_sigma_b: float = 0.06,
        generations: int = 80,
        seed: int = 42,
    ):
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutate_prob = mutate_prob
        self.mutate_sigma_w = mutate_sigma_w
        self.mutate_sigma_b = mutate_sigma_b
        self.generations = generations
        self.seed = seed

def next_generation(
    population: List[Policy],
    scores: np.ndarray,
    cfg: EvoConfig,
    rng: np.random.Generator,
) -> Tuple[List[Policy], Policy, float]:
    """Elitism + mutation; no crossover"""
    idx = np.argsort(scores)[::-1]
    elite_n = max(1, int(len(population) * cfg.elite_fraction))
    elites = [population[i].clone() for i in idx[:elite_n]]  # clone to freeze them
    best = elites[0]
    best_score = float(scores[idx[0]])

    new_pop: List[Policy] = []
    # keep elites
    new_pop.extend(elites)

    # fill rest with mutated copies sampled from elites
    while len(new_pop) < len(population):
        parent = elites[int(rng.integers(0, elite_n))]
        child = parent.clone()
        child.mutate(
            sigma_w=cfg.mutate_sigma_w,
            sigma_b=cfg.mutate_sigma_b,
            p=cfg.mutate_prob,
        )
        new_pop.append(child)

    return new_pop, best, best_score

def save_policy(policy: Policy, path: str):
    data = {
        "shapes": policy.shapes,
        "W": [w.tolist() for w in policy.W],
        "b": [b.tolist() for b in policy.b],
    }
    with open(path, "w") as f:
        json.dump(data, f)

def load_policy(path: str) -> Policy:
    with open(path, "r") as f:
        data = json.load(f)
    shapes = [tuple(s) for s in data["shapes"]]
    in_dim = shapes[0][0]
    out_dim = shapes[-1][1]
    hidden = tuple(s[1] for s in shapes[:-1])
    pol = Policy(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    pol.shapes = shapes
    pol.W = [np.array(w, dtype=np.float32) for w in data["W"]]
    pol.b = [np.array(b, dtype=np.float32) for b in data["b"]]
    return pol

# ---------------- features / obs ----------------

def next_pipe_for(bird_x: float, pipes: List[Pipe]) -> Pipe:
    if len(pipes) == 1:
        return pipes[0]
    p0 = pipes[0]
    if bird_x > p0.x + p0.PIPE_TOP.get_width():
        return pipes[1]
    return p0

def make_obs(bird: Bird, pipe: Pipe) -> np.ndarray:
    gap_mid = (pipe.height + pipe.bottom) / 2.0
    gap_half = (pipe.bottom - pipe.height) / 2.0
    bird_center_y = bird.y + bird.img.get_height() / 2.0
    dx = (pipe.x - bird.x)
    dy = (gap_mid - bird_center_y)
    vy = bird.vel
    return np.array([dx, dy, vy, gap_half], dtype=np.float32)

# --------------- single generation, visual ---------------

def evaluate_generation_visual(
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    font: pygame.font.Font,
    policies: List[Policy],
) -> np.ndarray:
    """Run one generation visually; return fitness per policy"""

    base = Base(FLOOR_Y)
    pipes = [Pipe(PIPE_SPAWN_X)]

    # per-bird state
    birds = []
    for pol in policies:
        birds.append({
            "policy": pol,
            "bird": Bird(230, 350),
            "alive": True,
            "fitness": 0.0,
            "counted": set(),  # pipe indices already credited
        })

    running = True
    while running:
        clock.tick(FPS)

        # basic event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        # move world first
        add_pipe = False
        remove_list = []

        for pipe in pipes:
            pipe.move()
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                remove_list.append(pipe)

        if pipes[-1].x < WIN_WIDTH - PIPE_ADD_GAP:
            pipes.append(Pipe(WIN_WIDTH))

        for rp in remove_list:
            pipes.remove(rp)

        base.move()

        # step all alive birds
        alive = 0
        for st in birds:
            if not st["alive"]:
                continue
            b = st["bird"]
            pol = st["policy"]

            # choose action from features vs this bird's next pipe
            p = next_pipe_for(b.x, pipes)
            obs = make_obs(b, p)
            action = pol.act(obs)
            if action == 1:
                b.jump()

            # physics
            b.move()

            # death by bounds
            if (b.y + b.img.get_height() - 10 >= FLOOR_Y) or (b.y < -50):
                st["alive"] = False
                continue

            # collision
            crashed = False
            for pipe in pipes:
                if pipe.collide(b):  # your signature is collide(self, bird)
                    st["alive"] = False
                    crashed = True
                    break
            if crashed:
                continue

            # living reward
            st["fitness"] += 0.05

            # pipe pass reward (per-bird)
            for i, pipe in enumerate(pipes):
                pipe_right = pipe.x + pipe.PIPE_TOP.get_width()
                if (pipe_right < b.x) and (i not in st["counted"]):
                    st["counted"].add(i)
                    st["fitness"] += 1.0

            alive += 1

        # end gen if none alive
        if alive == 0:
            running = False

        # render
        screen.fill((20, 20, 24))
        for pipe in pipes:
            pipe.draw(screen)

        shown = 0
        for st in birds:
            if not st["alive"]:
                continue
            if shown >= MAX_VISUAL_BIRDS:
                break
            # semi-transparent birds so overlaps are readable
            st["bird"].img.set_alpha(BIRD_ALPHA)
            st["bird"].draw(screen)
            shown += 1

        base.draw(screen)

        # HUD
        best_fit = max(s["fitness"] for s in birds) if birds else 0.0
        hud_lines = [
            f"Alive: {alive}/{len(birds)}",
            f"Gen best (this run): {best_fit:.1f}",
        ]
        y = 8
        for line in hud_lines:
            surf = font.render(line, True, (235, 235, 235))
            screen.blit(surf, (10, y))
            y += 26

        pygame.display.flip()

    # fitness array back to GA
    return np.array([st["fitness"] for st in birds], dtype=np.float32)

# ---------------- main visual evolution loop ----------------

def train_visual():
    clock = pygame.time.Clock()

    cfg = EvoConfig()
    rng = np.random.default_rng(cfg.seed)

    # bootstrap population
    population: List[Policy] = []
    for i in range(cfg.population_size):
        population.append(Policy(cfg.in_dim, cfg.hidden, cfg.out_dim))

    best_overall = None
    best_overall_score = -1e9

    gen = 1
    while gen <= cfg.generations:
        # show generation label quickly
        screen.fill((0, 0, 0))
        label = font.render(f"Generation {gen}", True, (255, 255, 255))
        screen.blit(label, (WIN_WIDTH//2 - label.get_width()//2, 12))
        pygame.display.flip()

        # evaluate visually
        scores = evaluate_generation_visual(screen, clock, font, population)

        # evolve
        population, best, best_score = next_generation(population, scores, cfg, rng)

        if best_score > best_overall_score:
            best_overall = best
            best_overall_score = best_score

        print(f"[Gen {gen:03d}] avg={scores.mean():.3f}  max={scores.max():.3f}  best_overall={best_overall_score:.3f}")

        # OPTIONAL: early stop if good enough
        # if best_overall_score >= 100.0:
        #     break

        gen += 1

    # save champion
    if best_overall is not None:
        save_policy(best_overall, "best_agent_visual.json")
        print("Saved best to best_agent_visual.json")

    pygame.quit()

if __name__ == "__main__":
    train_visual()
