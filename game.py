import pygame
import os

pygame.font.init()

# Setting the pygame window and game parameters
WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
START_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False

# Rendering the game window
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Importing game components
from components.bird import Bird
from components.base import Base
from components.pipes import Pipe

# Loading the backgroud image 
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("components/imgs","bg.png")).convert_alpha(), (600, 900))

def draw_game_window(win, birds, pipes, base, score):
    """Renders game components like - bird, base, pipes, background, current score onto the pygame window

    Args:
        win (pygame window): The game window
        birds (List[Bird]): List of Bird object(s) to be rendered
        pipes (List[Pipe]): List of Pipe Object(s) to be rendered
        base (Base): Base object to be rendered
        score (int): Current score to be rendered
    """
       
    win.blit(bg_img, (0, 0))
    
    for pipe in pipes:
        pipe.draw(win)
        
    base.draw(win)
    
    if len(birds) > 0:
        birds[0].draw(win)
    
    score_label = START_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))
    
    pygame.display.update()

def play_game():
    """Starts the Flappy Bird game and keeps it running till bird either goes out of bounds or collides with a pipe
    """
    
    global WIN      # Using global game window
    
    birds = []    # List of alive birds
    birds.append(Bird(230, 350))   # Initializing bird object
    
    base = Base(FLOOR)   # Initializing Base object
    pipes = [Pipe(700)]  # Initializing Pipe object
    score = 0   # Current score
    
    clock = pygame.time.Clock()   # Game clock
    
    run = True
    while run and len(birds) > 0:   # While the bird hasn't collided and there is atleast one bird alive
        clock.tick(30)   # Running the game at 30 FPS
        
        for event in pygame.event.get():    # Quit the game if window is closed
            if event.type == pygame.QUIT:   
                run = False
                pygame.quit()
                quit()
                break
            
            if event.type == pygame.KEYDOWN:   # make the bird jump if spacebar is pressed
                if event.key == pygame.K_SPACE:
                    for bird in birds:
                        bird.jump()
        
        for bird in birds:   # Move the bird(s)
            bird.move()  
        
        base.move()  # Move the pipe
        
        rem = []
        add_pipe = False    # Flag to check if new pipe needs to be added
        for pipe in pipes:
            pipe.move()     # Move all pipes on screen
            
            for bird in birds:
                if pipe.collide(bird=bird):   # Check for bird-pipe collision
                    birds.pop(birds.index(bird))
            
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
            
            if not pipe.passed and pipe.x < bird.x:   # if bird has passed the pipe, update 
                pipe.passed = True
                add_pipe = True
        
        if add_pipe:
            score += 1    # increment score if bird has passed a pipe
            
            pipes.append(Pipe(WIN_WIDTH))     
        
        for r in rem:
            pipes.remove(r)   
        
        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                birds.pop(birds.index(bird))
        
        draw_game_window(win=WIN, birds=birds, pipes=pipes, base=base, score=score) # rendering game window

if __name__ == '__main__':
    play_game()        