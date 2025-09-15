import sys
import json
import numpy as np
import pygame
import os

# Pygame parameters
WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
FPS = 30
PIPE_SPAWN = 700
PIPE_ADD_GAP = 300
HUD_FONT = "comicsans"

# Initializing pygame window
pygame.init()
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird Agent Evolution")
font = pygame.font.SysFont(HUD_FONT, 24)

from components.bird import Bird
from components.pipes import Pipe
from components.base import Base
from components.agent import Agent

# Loading the background image
bg_image = pygame.transform.scale(pygame.image.load(os.path.join("components/imgs", "bg.png")).convert_alpha(), (600, 900))

# Maximum number of birds to be displayed on the screen
MAX_VISUAL_BIRDS = 20

# Creating the Evolution Configuration class
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
        """Configuration for the agent

        Args:
            in_dim (int): Input dimensions (Here, 4 - [dx, dy, vy, gap_half])
            hidden (int): Dimensions of hidden layer
            out_dim (int): Output dimensions (Here, 1 - jump or not jump) 
            population_size (int): Number of agents to be trained per generation
            elite_fraction (float): Number of best agents to be directly cloned into the next generation
            mutate_prob (float): Probability of mutating the remaining agents
            mutate_sigma_w (float): The amount by which the weights should be altered.
            mutat_sigma_b (float): The amount by which the biases should be altered.
            generations (int): Number of generations to be trained for
            seed (int): Random seed for reproducibility
        """
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
    """Creating the new population for the next generation

    Args:
        population (List[Birds]): List of current population's birds
        scores (List[float]): Current population's fitness scores
        cfg (EvoConfig): Configurations for the agent population
        rng (np.random.Generator): Random number generator

    Returns:
        new_population(List[Bird]), best(Bird), best_score(float): new population of birds, best Bird from previous generation, best score from previous generation
    """
    idx = np.argsort(scores)[::-1]      # Sorting the scores in descending order 
    elite_n = max(1, int(len(population) * cfg.elite_fraction))         # Number of birds to directly be cloned
    elites = [population[i].clone() for i in idx[:elite_n]]         # List of birds to directly be cloned
    best = elites[0]        # best Bird from previous generation
    best_score = float(scores[idx[0]])      # best score from previous generation
    
    new_population = []         # Initializing empty list for new population
    new_population.extend(elites)       # Appending elite birds from previous generation into new one
    
    while len(new_population) < len(population):
        parent = elites[int(rng.integers(0, elite_n))]      #  Picking random bird from elites
        child = parent.clone()      # Cloning the parent
        child.mutate(       # Mutating the child
            sigma_w = cfg.mutate_sigma_w,   
            sigma_b = cfg.mutate_sigma_b,
            p = cfg.mutate_prob
        )
        new_population.append(child)        # Appending child into new population
        
    return new_population, best, best_score

def save_agent(agent, path):
    """Saving the agent in the path location in json format

    Args:
        agent (Agent): The bird to be saved
        path (str): Location to be stored in
    """
    # Storing the shapes, weights and biases of the given bird
    data = {
        "shapes": agent.shapes,
        "W": [w.tolist() for w in agent.W],
        "b": [b.tolist() for b in agent.b]
    }
    with open(path, "w") as f:
        json.dump(data, f)

def load_policy(path):
    """Loading the json file and initializing a Bird agent from the loaded weights, biases 

    Args:
        path (str): Path where json file is stored at
    """
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
    """Returns the next pipe to the right of bird's current x coordinate

    Args:
        bird_x (int): current x coordinate of the bird
        pipes (List[Pipe]): List of pipes on screen

    Returns:
        p0(Pipe): The next pipe immediately to the right of P
    """
    # If only a single pipe is on screen, it is the next to the right
    if len(pipes) == 1:
        return pipes[0]
    # A new pipe spwans only if bird has passed through previous, so if bird has passed through the 0th pipe, 1st pipe is the next
    p0 = pipes[0]
    if bird_x > p0.x + p0.PIPE_TOP.get_width():
        return pipes[1]
    return p0

def make_observation(bird, pipe):
    """Calculating input params for the agent

    Args:
        bird (Bird): Current Bird
        pipe (Pipe): Next pipe to it's immediate right

    Returns:
        List[float]: Input parameters for the agent
    """
    gap_mid = (pipe.height + pipe.bottom) / 2.0       # center of pipes' gap
    gap_half = (pipe.bottom - pipe.height) / 2.0        # gap between the two pipes
    bird_center_y = bird.y + bird.img.get_height() / 2.0    # y-coordinate of bird's center
    dx = (pipe.x - bird.x)      # x-distance between bird and pipe
    dy = (gap_mid - bird_center_y)      # y-distance between bird's center and pipe gap's center
    vy = bird.vel       # y-velocity of the bird
    return np.array([dx, dy, vy, gap_half], dtype=np.float32)

def evaluate_generation_visual(screen, clock, font, agents, generation):
    """Runs the simulation for a single generation

    Args:
        screen (pygame window): The pygame game window
        clock (pygame clock): The clock used for physics and animation calculation
        font (str): The font for the text 
        agents (List[Agent]): List of agents in current population
        generation (int): Current generation

    Returns:
        List[float]: Fitness scores of each agent in current population
    """
    base = Base(FLOOR)      # Initializing a Base object
    pipes = [Pipe(PIPE_SPAWN)]      # Initializing a list of pipes with a single pipe object
    
    birds = []      # Initializing empty birds list
    for agent in agents:
        birds.append({
            "agent": agent,     
            "bird": Bird(230, 350),
            "alive": True,
            "fitness": 0.0,
            "counted": set()    # set of pipes the bird has passed
        })
    
    running = True      # Is the game still running
    while running:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:       # quit on quitting the game window
                pygame.quit()
                sys.exit(0)
                
        remove_pipes_list = []      # Pipes to be removed from the original pipes list
        for pipe in pipes:
            pipe.move()     
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:      # If the bird passes the pipe, append it to pipes to be removed list
                remove_pipes_list.append(pipe)
        
        if pipes[-1].x < WIN_WIDTH - PIPE_ADD_GAP:      # Add a pipe if neccessary
            pipes.append(Pipe(WIN_WIDTH))
        
        for rem_pipe in remove_pipes_list:      # remove passed pipes 
            pipes.remove(rem_pipe)
            
        base.move()     # move the base
        
        alive = 0       # Number of alive birds
        for birb in birds:
            if not birb["alive"]:
                continue
            b = birb["bird"]
            ag = birb["agent"]
            
            p = next_pipe_for(bird_x=b.x, pipes=pipes)      # next pipe to the right
            observation = make_observation(bird=b, pipe=p)      
            action = ag.act(observation)        # agent's output
            if action == 1:
                b.jump()
            
            b.move()
            
            if (b.y + b.img.get_height() - 10 >= FLOOR) or (b.y < -50):     # check if bird has gone out of bounds
                birb["alive"]= False
                continue
            
            # check if the bird collided with any pipe
            crashed = False
            for pipe in pipes:
                if pipe.collide(bird=b):
                    birb["alive"] = False
                    crashed = True
                    break
                
                if crashed:
                    continue
            
            # Increase fitness for being alive
            birb["fitness"] += 0.05
            
            for i, pipe in enumerate(pipes):
                pipe_right = pipe.x + pipe.PIPE_TOP.get_width()
                if (pipe_right < b.x) and (i not in birb["counted"]):
                    birb["counted"].add(i)      # add passed pipes to 'counted'
                    birb["fitness"] += 1.0      # increase fitness if bird has passed a pipe
                    
            alive += 1
        
        # if all birds are dead, stop simulation
        if alive == 0:
            running = False
        
        screen.blit(bg_image, (0, 0))
        for pipe in pipes:
            pipe.draw(win=screen)
        
        # Display max 20 birds
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
            f"Current Generation: {generation}",
            f"Alive: {alive}/{len(birds)}",
            f"Gen best (this run): {best_fit:.1f}"
        ]
        
        y = 8
        for line in hud_lines:
            surf = font.render(line, True, (0, 0, 0)) 
            screen.blit(surf, (10, y))
            y += 26

        pygame.display.flip()
    
    return np.array([birb["fitness"] for birb in birds], dtype=np.float32)


def train_visual():
    """Run the entire training and visualization
    """
    clock = pygame.time.Clock()
    
    # Configuration ofr the agent
    cfg = EvoConfig(in_dim=4,
                    hidden=(32,32),
                    out_dim=1,
                    population_size=30,
                    elite_fraction=0.10,
                    mutate_prob=0.12,
                    mutate_sigma_w=0.12,
                    mutat_sigma_b=0.06,
                    generations=100,
                    seed=42)
    rng = np.random.default_rng(cfg.seed)
    
    # Creating the original population
    population = []
    for _ in range(cfg.population_size):        
        population.append(Agent(in_dim=cfg.in_dim, hidden=cfg.hidden, out_dim=cfg.out_dim))

    best_overall = None
    best_overall_score = -1e9
    
    gen = 1
    while gen <= cfg.generations:       
        label = font.render(f"Generation {gen}", True, (255, 255, 255))
        screen.blit(label, (WIN_WIDTH//2 - label.get_width()//2, 12))
        pygame.display.flip()
        
        scores = evaluate_generation_visual(screen=screen, clock=clock, font=font, agents=population, generation=gen)
        
        population, best, best_score = next_generation(population=population, scores=scores, cfg=cfg, rng=rng) 
        
        if best_score > best_overall_score:
            best_overall = best
            best_overall_score = best_score 
            save_agent(best_overall, "best_agent_visual.json")
            print("saved best to best_agent_visual.json")
            
        print(f"[Gen {gen:03d}] avg={scores.mean():.3f}  max={scores.max():.3f}  best_overall={best_overall_score:.3f}")

        gen += 1
        
    pygame.quit()
    
if __name__ == "__main__":
    train_visual()
        