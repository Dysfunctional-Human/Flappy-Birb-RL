import pygame
import os
import random

# Loading all the bird images (flap up, mid, down)
bird_images = [pygame.transform.scale(pygame.image.load(os.path.join("components/imgs", "bird" + str(x) + ".png")), (40,30)) for x in range(1,4)]

# Creating the Bird class
class Bird:
    MAX_ROTATION = 25    # Maximum upward rotation the bird can show
    IMGS = bird_images   # List of bird images
    ROT_VEL = 20         # Degree of rotation of bird 
    ANIMATION_TIME = 5   # Time for each sprite change of the bird
    
    def __init__(self, x, y):
        """Initializing a bird object

        Args:
            x (int): Initial x position of the bird
            y (int): Initial y position of the bird
        """
        self.x = x        
        self.y = y       
        self.tilt = 0    # Degrees by what the bird is rotated
        self.tick_count = 0   # Game tick - time since last jump (used for bird physics)
        self.vel = 0     # Vertical velocity of the bird
        self.height = self.y     # bird's y position at the time of last jump
        self.img_count = 0       # Current animation image of the bird
        self.img = self.IMGS[0]     # Current image of the bird
    
    def jump(self):
        """Makes the bird jump upwards
        """
        self.vel = -10.5     # Since top left of the window is 0,0 - going down is +ve, and going up is +ve
        self.tick_count = 0      # Resetting the tick count at the moment of jump
        self.height = self.y     # Setting current position as the y position since last jump
        
    def move(self):
        """Moves the bird by updating its position according to the game physics
        """
        self.tick_count += 1      #  Incrementing time since last jump
        
        displacement = (          # s = u*t + 0.5*a*t^2
            self.vel*self.tick_count +
            0.5*3*(self.tick_count**2)
        )
        
        if displacement >= 16:   # Limiting the max downward displacement for visual reasons
            displacement = 16
        
        self.y = self.y + displacement      # Updating bird's y position
        
        if displacement < 0 or self.y < self.height + 50:   # Making the bird point upwards while jumping
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION   
        elif self.tilt > -90:       # Making the bird vertically downwards in xase it overshoots
            self.tilt -= self.ROT_VEL
            
        
    def draw(self, win):
        """Renders the bird on the pygame window according to current animation and tilt

        Args:
            win (pygame window): The game window
        """
        self.img_count += 1
        
        # Making the bird flap its wings
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0
        
        # Setting flap down (angry birb mode) if bird is already falling
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2
        
        # Rotating the bird about it's center
        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)
        
    def get_mask(self):
        """Generates a rectangular mask of the bird's size

        Returns:
            pygame: Mask highlighting the pixels where the bird is present and 0 for transparent pixels
        """
        return pygame.mask.from_surface(self.img)

def blitRotateCenter(surf, image, topleft, angle):
    """Rotates the bird about it's center of rotation

    Args:
        surf (pygame window): the game window
        image (surface): the bird's surface
        topleft (Tuple(int, int)): coordinates of topleft point of the bird
        angle (int): Degree of rotation
    """
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)
    surf.blit(rotated_image, new_rect.topleft)