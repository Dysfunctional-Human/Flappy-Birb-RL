import pygame
import os
import random

# Loading the pipe image
pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("components/imgs","pipe.png")).convert_alpha())

# Creating the Pipe class
class Pipe:
    VEL = 5   # Horizontal velocity of pipes
    
    def __init__(self, x):
        """Initialize a Pipe object

        Args:
            x (int): Intial x position of the pipes
        """
        self.x = x   
        self.height = 0   # Height of top pipe
        
        self.gap = random.randint(135, 200)   # Gap between two pipes
        # self.gap = 220   # Gap between two pipes


        self.top = 0   # y position of starting of upper pipe
        self.bottom = 0   # y position of starting of lower pipe
        
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)  # Top pipe
        self.PIPE_BOTTOM = pipe_img  # Bottom pipe
        
        self.passed = False   # Has bird passed through the current pipes
        
        self.set_height()   # Update the upper and lower pipes' heights
        
    def set_height(self):
        """Setting the height of both - upper and lower pipes
        """
        self.height = random.randrange(50, 450)   # height of upper pipe
        self.top = self.height - self.PIPE_TOP.get_height()   # Starting point of upper pipe
        self.bottom = self.height + self.gap    # Starting point of lower pipe
    
    def move(self):
        """Shifting the two pipes to the left
        """
        self.x -= self.VEL
        
    def draw(self, win):
        """Rendering the two pipes on the game window

        Args:
            win (pygame window): the game window
        """
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))
        
    def collide(self, bird):
        """Checks whether the bird has collided with either of the two pipes

        Args:
            bird (Bird): the Bird

        Returns:
            bool: True if collision occured, false if not
        """
        bird_mask = bird.get_mask()   # gets the mask for bird
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)  # gets the mask for top pipe
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)   # gets the mask for bottom pipe
        top_offset = (self.x - bird.x, self.top - round(bird.y))   # Calculating relative distance between bird and top pipe
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))   # Calculating relative distance between bird and bottom pipe
        
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)    # Calculating overlap between bird and bottom pipe's masks based on their offset masks
        t_point = bird_mask.overlap(top_mask, top_offset)   # Same as above, returns first point of intersection between the two, None is no intersection
        
        if b_point or t_point:
            return True

        return False