import numpy as np
from PIL import Image
import cv2 as cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 5
HM_EPISODES = 25000
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
            self.y = SIZE-2
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
                        q_table[((i, ii), (iii, iiii))] = [[np.random.uniform(-8, 0),np.random.uniform(-8, 0)] for i in range(7)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example

episode_rewards = []

for episode in range(HM_EPISODES):
    package1 = Package(0,1)
    package2 = Package(SIZE-2,1)
    destination3 = Package(2, 0)
    destination1 = Package(SIZE-1, 0)
    destination2 = Package(1,0)
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    p1_out=False
    p2_out=False
    episode_reward = 0
    for i in range(1,2*SIZE**2):
        obs = (package1-destination1, package2-destination2)
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
        if (package1.x==0 or package1.y==SIZE-1 or package1.y==0 or package1.x==SIZE-1) and not p1_out:
            reward1= -OUT_PENALTY#*i*(abs(package1.x-destination1.x)+abs(package1.y-destination1.y))
            p1_got_out=True
        if ( package2.x==0 or package2.y==SIZE-1 or package2.y==0 or package2.x==SIZE-1) and not p2_out:    
            reward2= -OUT_PENALTY#*i*(abs(package2.x-destination2.x)+abs(package2.y-destination2.y))
            p2_got_out=True
        if package1.x == package2.x and package1.y == package2.y:
            if not p1_out and not p1_got_out:
                reward1-=COLLISION_PENALTY
            if not p2_out and not p2_got_out:
                reward2-=COLLISION_PENALTY
        else:
            if not p1_out and not p1_got_out:
                reward1-=MOVE_PENALTY
            if not p2_out and not p2_got_out:
                reward2-=MOVE_PENALTY
        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_obs = (package1-destination1, package2-destination2)
        max_future_q1 = np.max(q_table[new_obs],0)[0]
        current_q1 = q_table[obs][action1][0]

        if not p1_out:
            q_table[obs][action1][0] = (1 - LEARNING_RATE) * current_q1 + LEARNING_RATE * (reward1 + DISCOUNT * max_future_q1)
            reward+=reward1
       
        max_future_q2 = np.max(q_table[new_obs],0)[1]
        current_q2 = q_table[obs][action2][1]

        if not p2_out:
            q_table[obs][action2][1]= (1 - LEARNING_RATE) * current_q2 + LEARNING_RATE * (reward2 + DISCOUNT * max_future_q2)
            reward+=reward1
            
        
        
        if p1_got_out:
            p1_out=True
        if p2_got_out:
            p2_out=True
            
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            #for i in range(1,SIZE-1):
            #    for ii in range(1,SIZE-1):
            #        env[i][ii]=(255,255,255)
            env[destination1.y][destination1.x] = d[DESTINATION_N]  # sets the destination location tile to green color
           
            env[destination2.y][destination2.x] = d[PLAYER_N]  # sets the destination2 location to red
            env[destination3.y][destination3.x] = d[ENEMY_N]
            env[package1.y][package1.x] = d[DESTINATION_N]  # sets the package1 tile to blue
            env[package2.y][package2.x] = d[PLAYER_N]
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == 2*COLLISION_PENALTY or (p1_out and p2_out) :  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        #time.sleep(0.001)
        episode_reward += reward
        if reward == 2*COLLISION_PENALTY or (p1_out and p2_out):
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