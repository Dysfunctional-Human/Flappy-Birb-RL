import pygame
import os

# Loading the image of the base
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("components/imgs","base.png")).convert_alpha())

# Creating the Base class
class Base:
    VEL = 5   # Velocity of the moving base
    WIDTH = base_img.get_width()    # Width of the base's image
    IMG = base_img  
    
    def __init__(self, y):
        """Initializing a base object

        Args:
            y (int): y position of the base
        """
        self.y = y    
        self.x1 = 0     # Leftmost point of the first base image
        self.x2 = self.WIDTH     # Leftmost point of the second base image
        
    def move(self):
        """Moving the base
        """
        self.x1 -= self.VEL    # Moving the base to the left
        self.x2 -= self.VEL    
        if self.x1 + self.WIDTH < 0:     # Resetting the base if either of the two move out of the frame
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
    
    def draw(self, win):
        """Rendering the base onto the pygame window

        Args:
            win (pygame window): the game window
        """
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))
    