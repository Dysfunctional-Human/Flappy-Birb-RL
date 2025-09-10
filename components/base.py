import pygame
import os

# Loading the image of the base
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("components/imgs","base.png")).convert_alpha())

class Base:
    VEL = 5   # Velocity of the moving base
    WIDTH = base_img.get_width()    # Width of the base's image
    IMG = base_img  
    
    def __init__(self, y):
        self.y = y    
        self.x1 = 0
        self.x2 = self.WIDTH
        
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
    
    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))
    