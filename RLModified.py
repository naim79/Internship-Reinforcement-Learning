import numpy as np
from PIL import Image
import cv2 as cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 2
WRONGPLACE_PENALTY = 300
OUT_PENALTY= 300
DESTINATION_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 100  # how often to play through env visually.

start_q_table = None # None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # package key in dict
DESTINATION_N = 2  # destination key in dict
ENEMY_N = 3  # destination2 key in dict

# the dict!
d = {1: (255, 175, 0), #blue
     2: (0, 255, 0), #Green
     3: (0, 0, 255)} #Red

MOVE_OPTIONS=["EAST", "NORTHEAST", "NORTHWEST","WEST", "SOUTHWEST", "SOUTHEAST","STAY"]

class Package:
    def __init__(self, x=False, y=False):
        if not x:
            self.x = np.random.randint(1, SIZE-1)
            self.y = 8
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
            self.move((self.y)%2, 1)
        elif choice == 2:
            self.move( -(1+(self.y))%2, 1)
        elif choice == 3:
            self.move(-1, 0)
        elif choice == 4:
            self.move(-((self.y)+1)%2, -1)
        elif choice == 5:
            self.move((self.y)%2, -1)
        elif choice == 6:
            self.move(x=0, y=0)
            

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y


        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                    for iiii in range(-SIZE+1, SIZE):
                        q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-8, 0) for i in range(7)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example

episode_rewards = []

for episode in range(HM_EPISODES):
    package1 = Package(False,1)
    destination1 = Package(5, 0)
    destination3 = Package(9, 0)
    destination2 = Package(1,0)
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    
    episode_reward = 0
    for i in range(200):
        obs = (package1-destination1, package1-destination2)
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 7)
        # Take the action!
        package1.action(action)


        if (package1.x == destination2.x and package1.y == destination2.y) or (package1.x == destination3.x and package1.y == destination3.y):
            reward = -WRONGPLACE_PENALTY
        elif package1.x == destination1.x and package1.y == destination1.y:
            reward = DESTINATION_REWARD
        elif package1.x==0 or package1.y==SIZE or package1.y==0 or package1.x==SIZE:
            reward= -OUT_PENALTY
        else:
            reward = -MOVE_PENALTY
        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_obs = (package1-destination1, package1-destination2)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == DESTINATION_REWARD:
            new_q = DESTINATION_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            #for i in range(1,SIZE-1):
            #    for ii in range(1,SIZE-1):
            #        env[i][ii]=(255,255,255)
            env[destination1.y][destination1.x] = d[DESTINATION_N]  # sets the destination location tile to green color
            env[package1.y][package1.x] = d[PLAYER_N]  # sets the package1 tile to blue
            env[destination2.y][destination2.x] = d[ENEMY_N]  # sets the destination2 location to red
            env[destination3.y][destination3.x] = d[ENEMY_N]
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == DESTINATION_REWARD or reward == -WRONGPLACE_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        #time.sleep(0.001)
        episode_reward += reward
        if reward == DESTINATION_REWARD or reward == -WRONGPLACE_PENALTY or reward==-OUT_PENALTY:
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