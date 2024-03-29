import numpy as np
from PIL import Image
import cv2 as cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import pygame

style.use("ggplot")

SIZE = 7
HM_EPISODES =40000
MOVE_PENALTY = 0.5
OUT_PENALTY= 300
COLLISION_PENALTY=300
DESTINATION_REWARD = 200
epsilon = 0.9
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 100  # how often to play through env visually.

start_q_table = None # None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

map_file='map.png'
conv=pygame.image.load(map_file)
p1_file='greenPackage.png'
p1=pygame.image.load(p1_file)
p2_file='bluePackage.png'
p2=pygame.image.load(p2_file)
d1_file='greenCell.png'
d1=pygame.image.load(d1_file)
d2_file='blueCell.png'
d2=pygame.image.load(d2_file)
d3_file='redCell.png'
d3=pygame.image.load(d3_file)

screen_width = 700
screen_height = 580
screen = pygame.display.set_mode((screen_width, screen_height))

MOVE_OPTIONS=["EAST", "NORTHEAST", "NORTHWEST","WEST", "SOUTHWEST", "SOUTHEAST","STAY"]

class Cell:
    def __init__(self, x=False, y=False):
        if not x and not y:
            self.x = np.random.randint(1, SIZE-1)
            self.y = 1
        else:
            self.x=x
            self.y=y

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        '''
        ["EAST", "NORTHEAST", "NORTHWEST","WEST", "SOUTHWEST", "SOUTHEAST","STAY"]
        '''
        if choice == 0:
            self.move(1, 0)
        elif choice == 1:
            self.move((self.y+1)%2, 1)
        elif choice == 2:
            self.move( -((self.y)%2), 1)
        elif choice == 3:
            self.move(-1, 0)
        elif choice == 4:
            self.move(-((self.y)%2), -1)
        elif choice == 5:
            self.move((self.y+1)%2, -1)
        elif choice == 6:
            self.move(0, 0)
            

    def move(self, x, y):
        self.x += x
        self.y += y

    def toPair(self):
        return (self.x, self.y)
    
if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for i in range(SIZE):
        for ii in range(SIZE):
            for iii in range(SIZE):
                    for iiii in range(SIZE):
                        q_table[((i, ii), (iii, iiii))] = [[np.random.uniform(-8, 0),np.random.uniform(-8, 0)] for i in range(7)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example

episode_rewards = []

for episode in range(HM_EPISODES):
    while True:
        package1 = Cell()
        package2 = Cell()
        if not package1.x-package2.x==0:
            break
    destination1 = Cell(3, SIZE-1)
    destination2 = Cell(SIZE-1,3)
    destination3 = Cell(0, 3)
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    p1_out=False
    p2_out=False
    episode_reward = 0
    if show:
            screen.blit(conv, (0, 0))
            screen.blit(d1, (destination1.x*100+50*((destination1.y+1)%2), 580-20-(destination1.y+1)*80))
            screen.blit(d2, (destination2.x*100+50*((destination2.y+1)%2), 580-20-(destination2.y+1)*80))
            screen.blit(d3, (destination3.x*100+50*((destination3.y+1)%2), 580-20-(destination3.y+1)*80))
            screen.blit(p1, (100*package1.x+50*((package1.y+1)%2)+10, 580-((package1.y+1)*80)-10))
            screen.blit(p2, (100*package2.x+50*((package2.y+1)%2)+10, 580-((package2.y+1)*80)-10))
            pygame.display.flip()
            #time.sleep(0.5)
    for i in range(1,2*SIZE**2):
        obs = (package1.toPair(), package2.toPair())
        copyP1=package1
        copyP2=package2
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action1 = np.argmax(q_table[obs],0)[0]
            action2 = np.argmax(q_table[obs],0)[1]
        else:
            action1 = np.random.randint(0, 7)
            action2 = np.random.randint(0, 7)
        # Take the action!
        if not p1_out:
            package1.action(action1)
        if not p2_out:
            package2.action(action2)
        
        reward1=0
        reward2=0
        reward=0
        p1_got_out=False
        p2_got_out=False
        if package1.x == destination1.x and package1.y == destination1.y and not p1_out:
            reward1 = DESTINATION_REWARD
            q_table[obs][action1][0] =reward1
            p1_out=True
            reward+=reward1
        if package2.x == destination2.x and package2.y == destination2.y and not p2_out:
            reward2 = DESTINATION_REWARD
            q_table[obs][action2][1] =reward2
            p2_out=True
            reward+=reward2
        if (package1.x==0 or package1.y==SIZE-1 or package1.y==0 or package1.x==SIZE-1 or (package1.x==SIZE-2 and package1.y%2==0)) and not p1_out:
            reward1= -OUT_PENALTY#*i*(abs(package1.x-destination1.x)+abs(package1.y-destination1.y))
            p1_got_out=True
        if ( package2.x==0 or package2.y==SIZE-1 or package2.y==0 or package2.x==SIZE-1 or (package2.x==SIZE-2 and package2.y%2==0)) and not p2_out:    
            reward2= -OUT_PENALTY#*i*(abs(package2.x-destination2.x)+abs(package2.y-destination2.y))
            p2_got_out=True
        if package1.x == package2.x and package1.y == package2.y:
            if not p1_out and not p1_got_out:
                reward1-=COLLISION_PENALTY
                reward2-=COLLISION_PENALTY
        elif (package1-copyP2)==(0,0) and (package2-copyP1)==(0,0):
            reward1-=COLLISION_PENALTY
            reward2-=COLLISION_PENALTY
        else:
            if not p1_out and not p1_got_out:
                reward1-=MOVE_PENALTY
            if not p2_out and not p2_got_out:
                reward2-=MOVE_PENALTY
        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_obs = (package1.toPair(), package2.toPair())
        max_future_q1 = np.max(q_table[new_obs],0)[0]
        current_q1 = q_table[obs][action1][0]

        if not p1_out:
            q_table[obs][action1][0] = (1 - LEARNING_RATE) * current_q1 + LEARNING_RATE * (reward1 + DISCOUNT * max_future_q1)
            reward+=reward1
       
        max_future_q2 = np.max(q_table[new_obs],0)[1]
        current_q2 = q_table[obs][action2][1]

        if not p2_out:
            q_table[obs][action2][1]= (1 - LEARNING_RATE) * current_q2 + LEARNING_RATE * (reward2 + DISCOUNT * max_future_q2)
            reward+=reward2
            
        if p1_got_out:
            p1_out=True
        if p2_got_out:
            p2_out=True
            
        if show:
            screen.blit(conv, (0, 0))
            screen.blit(d1, (destination1.x*100+50*((destination1.y+1)%2), 580-20-(destination1.y+1)*80))
            screen.blit(d2, (destination2.x*100+50*((destination2.y+1)%2), 580-20-(destination2.y+1)*80))
            screen.blit(d3, (destination3.x*100+50*((destination3.y+1)%2), 580-20-(destination3.y+1)*80))
            screen.blit(p1, (100*package1.x+50*((package1.y+1)%2)+10, 580-((package1.y+1)*80)-10))
            screen.blit(p2, (100*package2.x+50*((package2.y+1)%2)+10, 580-((package2.y+1)*80)-10))
            pygame.display.flip()
            #time.sleep(0.5)
        episode_reward += reward
        if reward <-3 or (p1_out and p2_out) or p1_got_out or p2_got_out:
            break
        
    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
