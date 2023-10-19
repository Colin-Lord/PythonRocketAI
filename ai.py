import math
import pygame
import random

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1920, 1080))
pygame.display.set_caption('AI Rockets')
clock = pygame.time.Clock()
running = True
dt = 0

generation = 0


#sets up lists of variables for position, rotation, etc. in this case, rotation is not going to use momentum or forces
acceleration = [pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0)]
velocity = [pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0)]
position = [pygame.Vector2(screen.get_width() / 2, 100),pygame.Vector2(screen.get_width() / 2, 100),pygame.Vector2(screen.get_width() / 2, 100),pygame.Vector2(screen.get_width() / 2, 100),pygame.Vector2(screen.get_width() / 2, 100),pygame.Vector2(screen.get_width() / 2, 100),pygame.Vector2(screen.get_width() / 2, 100),pygame.Vector2(screen.get_width() / 2, 100),pygame.Vector2(screen.get_width() / 2, 100),pygame.Vector2(screen.get_width() / 2, 100)]

rotation = [0,0,0,0,0,0,0,0,0,0] #uses degrees
thrust = [0,0,0,0,0,0,0,0,0,0]

alive = [1,1,1,1,1,1,1,1,1,1]

thrustRate = 1000
mutationRate = 0.5

directionalThrust = [pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0),pygame.Vector2(0, 0)]


highestFitness = 0
fitness = [0,0,0,0,0,0,0,0,0,0]

initalRandomRange = 5

#sets up the neural network weights, 10 rockets, 21 different weights
rows, cols = (10, 21)
NeuralNetworkWeights = [[0] * cols for _ in range(rows)]
BestWeights = [[0] * cols for _ in range(rows)]


#load rocket
rocket = [pygame.image.load("images/rocket.png"),pygame.image.load("images/rocket.png"),pygame.image.load("images/rocket.png"),pygame.image.load("images/rocket.png"),pygame.image.load("images/rocket.png"),pygame.image.load("images/rocket.png"),pygame.image.load("images/rocket.png"),pygame.image.load("images/rocket.png"),pygame.image.load("images/rocket.png"),pygame.image.load("images/rocket.png")]
rocketRectangle = rocket[0].get_rect()

#load background
backgroundImg = pygame.image.load("images/background.jpg")





def initialGeneration():
    g = 0
    while g < 10:

        P = 0
        while P < 21:
            NeuralNetworkWeights[g][P] = random.uniform(-1, 1)
            P += 1
        g += 1

##########################

def draw_img(image, x, y, angle):
    rotated_image = pygame.transform.rotate(image, angle) 
    screen.blit(rotated_image, rotated_image.get_rect(center=image.get_rect(topleft=(x, y)).center).topleft)


##########################

def KillRocket(index):
    alive[index] = 0

##########################




##########################

def SpawnRockets():
    i = 0
    while i < 10:
        alive[i] = 1
        position[i] = pygame.Vector2(screen.get_width() / 2, 100)
        velocity[i] = pygame.Vector2(0, 0)
        acceleration[i] = pygame.Vector2(0, 0)

        i += 1  

##########################




##########################

def SaveBestWeights(index):

    for x in BestWeights:
        BestWeights[x] = NeuralNetworkWeights[index][x]

##########################




##########################

def Mutate(index):
    for x in BestWeights:
        NeuralNetworkWeights[index][x] = BestWeights[x] + (random.uniform(-1, 1) * mutationRate)

##########################




##########################

def InitialRandom(index):
    for x in NeuralNetworkWeights:
        NeuralNetworkWeights[index][x] = random.uniform(-initalRandomRange, initalRandomRange)

##########################


def sigmoid(x):
    x = max(-700, min(700, x))  # Clip x to a reasonable range to prevent overflow
    return 1 / (1 + math.exp(-x))


##########################

def NeuralNetwork(angle, position2d, velocity2d, rocketindex):

    hiddenlayer1a = angle * NeuralNetworkWeights[rocketindex][0] + position2d.x * NeuralNetworkWeights[rocketindex][3] + position2d.y * NeuralNetworkWeights[rocketindex][6] + velocity2d.x * NeuralNetworkWeights[rocketindex][9] + velocity2d.y * NeuralNetworkWeights[rocketindex][12]
    hiddenlayer1b = angle * NeuralNetworkWeights[rocketindex][1] + position2d.x * NeuralNetworkWeights[rocketindex][4] + position2d.y * NeuralNetworkWeights[rocketindex][7] + velocity2d.x * NeuralNetworkWeights[rocketindex][10] + velocity2d.y * NeuralNetworkWeights[rocketindex][13]
    hiddenlayer1c = angle * NeuralNetworkWeights[rocketindex][2] + position2d.x * NeuralNetworkWeights[rocketindex][5] + position2d.y * NeuralNetworkWeights[rocketindex][8] + velocity2d.x * NeuralNetworkWeights[rocketindex][11] + velocity2d.y * NeuralNetworkWeights[rocketindex][14]

    hiddenlayer1a = sigmoid(hiddenlayer1a)
    hiddenlayer1b = sigmoid(hiddenlayer1b)
    hiddenlayer1c = sigmoid(hiddenlayer1c)

    output1 = hiddenlayer1a * NeuralNetworkWeights[rocketindex][15] + hiddenlayer1b * NeuralNetworkWeights[rocketindex][17] + hiddenlayer1c * NeuralNetworkWeights[rocketindex][19]
    output2 = hiddenlayer1a * NeuralNetworkWeights[rocketindex][16] + hiddenlayer1b * NeuralNetworkWeights[rocketindex][18] + hiddenlayer1c * NeuralNetworkWeights[rocketindex][20]

    rotation[rocketindex] = sigmoid(rotation[rocketindex])
    thrust[rocketindex] = sigmoid(thrust[rocketindex])

    rotation[rocketindex]  = rotation[rocketindex] * 360
    thrust[rocketindex] = thrust[rocketindex] * thrustRate
##########################

initialGeneration()

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    screen.blit(backgroundImg, (0, 0))

    result = False

    for x in alive:
        if x == 0:
            result = True
        else: break
    
    if result:
        SpawnRockets()
        generation += 1


    #neural network
    i = 0
    while i < 10:
        NeuralNetwork(rotation[i], position[i], velocity[i], i)
        i += 1

    #calculate position of rockets
    i = 0
    while i < 10:

        directionalThrust[i] = pygame.Vector2(thrust[i] * math.cos(rotation[i]), thrust[i] * math.cos(rotation[i])) * dt

        acceleration[i].x += directionalThrust[i].x * dt
        acceleration[i].y += directionalThrust[i].y * dt

        #gravity
        acceleration[i].y += 2000 * dt

        velocity[i].x += acceleration[i].x * dt
        velocity[i].y += acceleration[i].y * dt

        position[i].x += velocity[i].x * dt
        position[i].y += velocity[i].y * dt

        i += 1


    #kills the rockets if they get too close to the edge, with a 10 pixel padding around the edge
    i = 0
    while i < 10:
        if position[i].x < 10 or position[i].x > 1910:
            alive[i] = 0
        if position[i].y < 10 or position[i].y > 1070:
            alive[i] = 0
        i += 1

    #draw all of the rockets
    i = 0
    while i < 10:

        if alive[i] == 1:
            draw_img(rocket[i], position[i].x, position[i].y, rotation[i])
            

        i += 1

    
    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(144) / 1000

pygame.quit()