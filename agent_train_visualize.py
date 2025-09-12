import sys
import json
import numpy as np
import pygame
import os

from typing import List, Tuple

# Pygame parameters
WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
FPS = 30
PIPE_SPAWN = 700
PIPE_ADD_GAP = 300
HUD_FONT = "comicsans"

pygame.init()
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird Agent Evolution")
font = pygame.font.SysFont(HUD_FONT, 24)

from components.bird import Bird
from components.pipes import Pipe
from components.base import Base
from components.agent import Agent

bg_image = pygame.transform.scale(pygame.image.load(os.path.join("components/imgs", "bg.png")).convert_alpha(), (600, 900))

MAX_VISUAL_BIRDS = 20

class EvoConfig:
    def __init__(self,
                 in_dim,
                 hidden,
                 out_dim,
                 population_size,
                 elite_fraction,
                 mutate_prob,
                 mutate_sigma_w,
                 mutat_sigma_b,
                 generations,
                 seed):
        
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutate_prob = mutate_prob
        self.mutate_sigma_w = mutate_sigma_w
        self.mutate_sigma_b = mutat_sigma_b
        self.generations = generations
        self.seed = seed
        
def next_generation(population, scores, cfg, rng: np.random.Generator):
    idx = np.argsort(scores)[::-1]
    elite_n = max(1, int(len(population) * cfg.elite_fraction))
    elites = [population[i].clone() for i in idx[:elite_n]]
    best = elites[0]
    best_score = float(scores[idx[0]])
    
    new_population = []
    new_population.extend(elites)
    
    while len(new_population) < len(population):
        parent = elites[int(rng.integers(0, elite_n))]
        child = parent.clone()
        child.mutate(
            sigma_w = cfg.mutate_sigma_w,   
            sigma_b = cfg.mutate_sigma_b,
            p = cfg.mutate_prob
        )
        new_population.append(child)
        
    return new_population, best, best_score

def save_agent(agent, path):
    data = {
        "shapes": agent.shapes,
        "W": [w.tolist() for w in agent.W],
        "b": [b.tolist() for b in agent.b]
    }
    with open(path, "w") as f:
        json.dump(data, f)

def load_policy(path):
    with open(path, "r") as f:
        data = json.load(f)
    shapes = [tuple(s) for s in data["shapes"]]
    in_dim = shapes[0][0]
    out_dim = shapes[-1][1]
    hidden = tuple(s[1] for s in shapes[:-1])
    agent = Agent(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    agent.shapes = shapes
    agent.W = [np.array(w, dtype=np.float32) for w in data["W"]]
    agent.b = [np.array(b, dtype=np.float32) for b in data["b"]]
    
def next_pipe_for(bird_x, pipes):
    if len(pipes) == 1:
        return pipes[0]
    p0 = pipes[0]
    if bird_x > p0.x + p0.PIPE_TOP.get_width():
        return pipes[1]
    return p0

def make_observation(bird, pipe):
    gap_mid = (pipe.height + pipe.bottom) / 2.0
    gap_half = (pipe.bottom - pipe.height) / 2.0
    bird_center_y = bird.y + bird.img.get_height() / 2.0
    dx = (pipe.x - bird.x)
    dy = (gap_mid - bird_center_y)
    vy = bird.vel
    return np.array([dx, dy, vy, gap_half], dtype=np.float32)

def evaluate_generation_visual(screen, clock, font, agents):
    base = Base(FLOOR)
    pipes = [Pipe(PIPE_SPAWN)]
    
    birds = []
    for agent in agents:
        birds.append({
            "agent": agent,
            "bird": Bird(230, 350),
            "alive": True,
            "fitness": 0.0,
            "counted": set()
        })
    
    running = True
    while running:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
                
        remove_pipes_list = []
        for pipe in pipes:
            pipe.move()
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                remove_pipes_list.append(pipe)
        
        if pipes[-1].x < WIN_WIDTH - PIPE_ADD_GAP:
            pipes.append(Pipe(WIN_WIDTH))
        
        for rem_pipe in remove_pipes_list:
            pipes.remove(rem_pipe)
            
        base.move()
        
        alive = 0
        for birb in birds:
            if not birb["alive"]:
                continue
            b = birb["bird"]
            ag = birb["agent"]
            
            p = next_pipe_for(bird_x=b.x, pipes=pipes)
            observation = make_observation(bird=b, pipe=p)
            action = ag.act(observation)
            if action == 1:
                b.jump()
            
            b.move()
            
            if (b.y + b.img.get_height() - 10 >= FLOOR) or (b.y < -50):
                birb["alive"]= False
                continue
            
            crashed = False
            for pipe in pipes:
                if pipe.collide(bird=b):
                    birb["alive"] = False
                    crashed = True
                    break
                
                if crashed:
                    continue
            
            birb["fitness"] += 0.05
            
            for i, pipe in enumerate(pipes):
                pipe_right = pipe.x + pipe.PIPE_TOP.get_width()
                if (pipe_right < b.x) and (i not in birb["counted"]):
                    birb["counted"].add(i)
                    birb["fitness"] += 1.0
                    
            alive += 1
            
        if alive == 0:
            running = False
        
        screen.blit(bg_image, (0, 0))
        for pipe in pipes:
            pipe.draw(win=screen)
        
        shown = 0
        for birb in birds:
            if not birb["alive"]:
                continue
            if shown >= MAX_VISUAL_BIRDS:
                break
            
            birb["bird"].draw(win=screen)
            shown += 1
        
        base.draw(win=screen)
        
        best_fit = max(s["fitness"] for s in birds) if birds else 0.0
        hud_lines = [
            f"Alive: {alive}/{len(birds)}",
            f"Gen best (this run): {best_fit:.1f}"
        ]
        
        y = 8
        for line in hud_lines:
            surf = font.render(line, True, (235, 235, 235))
            screen.blit(surf, (10, y))
            y += 26

        pygame.display.flip()
    
    return np.array([birb["fitness"] for birb in birds], dtype=np.float32)


def train_visual():
    clock = pygame.time.Clock()
    
    cfg = EvoConfig(in_dim=4,
                    hidden=(32,32),
                    out_dim=1,
                    population_size=25,
                    elite_fraction=0.10,
                    mutate_prob=0.12,
                    mutate_sigma_w=0.12,
                    mutat_sigma_b=0.06,
                    generations=80,
                    seed=42)
    rng = np.random.default_rng(cfg.seed)
    
    population = []
    for _ in range(cfg.population_size):        
        population.append(Agent(in_dim=cfg.in_dim, hidden=cfg.hidden, out_dim=cfg.out_dim))

    best_overall = None
    best_overall_score = -1e9
    
    gen = 1
    while gen <= cfg.generations:
        label = font.render(f"Genetation {gen}", True, (255, 255, 255))
        screen.blit(label, (WIN_WIDTH//2 - label.get_width()//2, 12))
        pygame.display.flip()
        
        scores = evaluate_generation_visual(screen=screen, clock=clock, font=font, agents=population)
        
        population, best, best_score = next_generation(population=population, scores=scores, cfg=cfg, rng=rng) 
        
        if best_score > best_overall_score:
            best_overall = best
            best_overall_score = best_score 
            
        print(f"[Gen {gen:03d}] avg={scores.mean():.3f}  max={scores.max():.3f}  best_overall={best_overall_score:.3f}")

        gen += 1
        
    if best_overall is not None:
        save_agent(best_overall, "best_agent_visual.json")
        print("saved best tp best_agent_visual.json")
        
    pygame.quit()
    
if __name__ == "__main__":
    train_visual()
        