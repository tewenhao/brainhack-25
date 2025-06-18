import numpy as np
import gymnasium as gym
from til_environment import gridworld
class gridworldRevampedEnv(gym.Env):
    def __init__(self):
        self.n_players = 4
        self.current_player_num = 0
        self.flatten_obs = True
        self.offset_direction = ((-1,-1),(1,-1),(1,1),(-1,1))
        self.size =16
        self.env = gridworld.env(env_wrappers = [],render_mode = None,novice = False)
    def reset(self):
        """resets environment, returns observation for first player"""
        self.current_player_num = 0
        self.env.reset()
        self.collated_map = []
        self.collated_sightings = []
        for i in range(self.n_players):
            self.collated_map.append(np.zeros(shape=(16,16,6)))
            self.collated_sightings.append(np.zeros(shape=(3,16,16)))
    def change_direction(self,x,y,direction,direction_forward,direction_down):
        x = int(x)
        y = int(y)
        if direction == 0:
            x += direction_forward
            y += direction_down
        elif direction == 1:
            x -= direction_down
            y += direction_forward
        elif direction == 2:
            x -= direction_forward
            y -= direction_down
        elif direction == 3:
            x += direction_down
            y -= direction_forward
        return int(x),int(y)
    def step(self,action):
        """Takes one action, returns the reward for that action, and if the game is done"""
        #Only if the current agent is the last agent should we reset the agent selector
        #And renew the observations
        self.env.step(action)
        self.current_player_num = (self.current_player_num +1)%4
        #observation, reward, termination, truncation, info = self.env.last()

        pass
        return observation,reward,done,{}
    def render(self):
        pass

    def observation(self):
        """Returns observations for the current player"""

        # if self.flatten_obs:
        self.update_map(self,self.current_player_num,observation)
        return observation
    def update_map(self,observation):
        viewcone = observation["viewcone"]
        viewcone = np.rot90(viewcone, k=-1)
        viewcone = np.fliplr(viewcone)
        self.location = observation["location"]
        self.permanent_value[self.location[1],self.location[0]] -=1
        x,y = self.location
        x,y = int(x),int(y)
        direction = observation["direction"]
        viewcone_length = len(viewcone[0])
        viewcone_width = len(viewcone)
        initial_offset = 2
        offsetx = x
        offsety = y
        offsetchange = self.offset_direction[direction]
        offsetx += offsetchange[0]*initial_offset
        offsety += offsetchange[1]*initial_offset
        for i in range(0,viewcone_length):
            for j in range(0,viewcone_width):
                # print(i,j,end="|")
                currentx,currenty = self.change_direction(offsetx,offsety,direction,i,j)
                if currentx< 0 or currentx >=self.size or currenty < 0 or currenty >= self.size:
                    continue
                    raise Exception(f"Out of bounds {currentx},{currenty}")
                # print(i,j)
                info = viewcone[j][i]
                if info == 0 :
                    continue
                vision = info&3
                if vision == 1:
                    self.value_map[currenty][currentx] = 0
                elif vision == 2:
                    self.value_map[currenty][currentx] = 1
                elif vision == 3:
                    self.value_map[currenty][currentx] = 5
                if self.get_bit(info,3) and self.is_scout:
                    self.add_last_known_position((currentx,currenty))
                
                for k in range(4,8):
                    wall_direction = (k+int(direction))%4#(k-4+1+int(direction))%4
                    has_wall = self.get_bit(info,k)
                    
                    self.wall_map[currenty][currentx][wall_direction] =has_wall
                    otherside_x,otherside_y = self.change_direction(currentx,currenty, wall_direction,1,0)
                    if self.validate_location((otherside_x,otherside_y)):
                        self.wall_map[otherside_y][otherside_x][(wall_direction+2)%4] =has_wall
                # print("wallmap",np.binary_repr(info, width=8),self.wall_map[currenty][currentx],direction)
                # if currentx == x and currenty == y:
                    # print(self.wall_map[currenty][currentx])

    def legal_actions(self):
        return self.