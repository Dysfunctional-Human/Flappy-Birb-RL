import pygame
import os

pygame.font.init()

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
START_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

from components.bird import Bird
from components.base import Base
from components.pipes import Pipe
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("components/imgs","bg.png")).convert_alpha(), (600, 900))

def draw_regular_game_window(win, birds, pipes, base, score, pipe_ind):
    win.blit(bg_img, (0, 0))
    
    for pipe in pipes:
        pipe.draw(win)
        
    base.draw(win)
    
    if len(birds) > 0:
        birds[0].draw(win)
    
    score_label = START_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))
    
    pygame.display.update()

def play_regular_game():
    global WIN
    win = WIN
    
    birds = []
    birds.append(Bird(230, 350))
    
    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0
    
    clock = pygame.time.Clock()
    
    run = True
    while run and len(birds) > 0:
        clock.tick(30)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    for bird in birds:
                        bird.jump()
        
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        
        for bird in birds:
            bird.move()
        
        base.move()
        
        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            
            for bird in birds:
                if pipe.collide(bird=bird, win=win):
                    birds.pop(birds.index(bird))
            
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
            
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True
        
        if add_pipe:
            score += 1
            
            pipes.append(Pipe(WIN_WIDTH))
        
        for r in rem:
            pipes.remove(r)
        
        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                birds.pop(birds.index(bird))
        
        draw_regular_game_window(win=WIN, birds=birds, pipes=pipes, base=base, score=score, pipe_ind=pipe_ind)

if __name__ == '__main__':
    play_regular_game()        