import numpy as np
import copy
from collections import deque
# import pygame
# import gridworld3
import matplotlib.pyplot as plt
import cv2 as cv
import time
import gridworld
np.set_printoptions(precision=1)
class Scout():
    def __init__(self,size = 16,is_scout= True,first=False,local = True,map_generator=None):   
        self.local = local
        if not self.local:
            self.map_generator = map_generator
        self.location = [0,0]
        self.size = size
        self.value_map = np.full(shape=(size,size),dtype=np.uint8,fill_value=2)
        #each element represents the value ie nromal or challenge or unknown(which has the same value as none)
        self.last_known_location = []
        #index 0 is right, 1 is down 2 is left 3 is up
        self.direction_transmutation = ((0,1),(1,2),(2,3),(3,0))
        self.offset_direction = ((-1,-1),(1,-1),(1,1),(-1,1))
        # # self.env = gridworld3.env()
        self.max_length = 10
        # for i in range(3):
        #     self.last_known_location.append([-1,-1])
        self.enemy_penalty = -50
        self.chosen_possible = -1 
        self.gamma = 0.9
        self.enemy_gamma = 0.7
        self.old_location = []
        self.first = first
        self.is_scout=is_scout
        if not is_scout:
            self.loc_target = [[3,3],[3,13],[13,3]]
            self.loc_target_copy = [[3,3],[3,13],[13,3]]
            self.loc_target_index = 0
            self.possible_location =  [[4,4],[4,12],[12,12],[12,4]]
        self.render_mode = "human" if first else ""
        self.window = None
        window_size = 768
        self.debug=False
        self.window_size = window_size  # vertical size of the PyGame window
        self.window_width = int(window_size * 1.5) if self.debug else window_size
        self.window = None
        self.clock = None
        self.font = None
        self.step = 0
        self.last_seen_guard = []
        self.reset()
    def reset(self):
        self.reverse = False
        self.wall_map = np.zeros(shape= (self.size,self.size,4),dtype=np.uint8)#same 0 is right, 1 is down, 2 is left , 3 is up
        self.permanent_value = np.zeros(shape= (self.size,self.size),dtype=np.int32)
    def direction_to_array(self,dir):
        return ((1,0),(0,1),(-1,0),(0,-1))[dir]
    def validate_location(self,location):
        if location[0]< 0 or location[0] >=self.size or location[1] < 0 or location[1] >= self.size:
            return False
        return True
        raise Exception(f"Out of bounds {location}")
    def set_distance_map(self,location,distance):
        self.distance_map[location[1]][location[0]] = distance
    def calculate_value_map(self):
        self.accumulated_value_map = np.full(shape=(self.size,self.size),fill_value=-1,dtype=np.float32)
        self.distance_map = np.full(shape=(self.size,self.size),fill_value=-1,dtype=np.int8)#np.zeros(shape=(self.size,self.size),dtype=np.uint8)
        distance_queue = deque()
        distance_queue.append((0,tuple(self.location)))
        distance_array = []
        calls = 0
        for i in range(self.size**2):
            distance_array.append([])
        while distance_queue:
            calls +=1
            
            current_info = distance_queue.popleft()
            # print(current_info)
            distance,location = current_info
            if self.distance_map[location[1]][location[0]]!=-1:
                continue
            self.set_distance_map(location,distance)
            walls = self.wall_map[location[1]][location[0]]
            for possible_direction,valid in enumerate(walls):
                if valid:
                    continue
                check_location = self.change_direction(location[0],location[1],possible_direction,1,0)
                if not self.validate_location(check_location):
                    continue
                if self.distance_map[check_location[1]][check_location[0]]==-1:
                    distance_array[distance].append(check_location)
                    distance_queue.append((distance+1,check_location))
        # for 
        # locat
        # self.distance_map[self.location[1]][self.location[0]] = 0
        # print(self.distance_map)
        for distance in range(31,-1,-1):
            dist_arr = distance_array[distance]
            for location in dist_arr:
                value = self.value_map[location[1],location[0]]
                walls = self.wall_map[location[1]][location[0]]
                max_value = 0 
                for possible_direction,not_valid in enumerate(walls):
                    if not_valid:
                        continue
                    adjx,adjy= self.change_direction(location[0],location[1],possible_direction,1,0)
                    if not self.validate_location((adjx,adjy)):
                        continue
                    if self.distance_map[adjy,adjx]>distance:
                        if self.gamma*self.accumulated_value_map[adjy,adjx] > max_value:
                            max_value = self.gamma*self.accumulated_value_map[adjy,adjx]
                            
                self.accumulated_value_map[location[1],location[0]] = value + max_value
        # print(self.accumulated_value_map)
        # print("gay")

    def calculate_guards_danger(self):
        self.current_enemy_map = np.full(shape=(self.size,self.size),fill_value=0,dtype=np.float32)
        for enemy_location in self.last_known_location:
            self.guard_bfs(enemy_location)
        
    def guard_bfs(self,enemy_location):
        # for now i assume last seen position doesnt change
        next_enemy_map=np.full(shape=(self.size,self.size),fill_value=0,dtype=np.float32)
        bfsqueue = deque()
        bfsqueue.append(enemy_location)
        probability = 1
        number = 0
        bfsqueuenext = deque()
        while bfsqueue:
            while bfsqueue:
                location = bfsqueue.popleft()
                if next_enemy_map[location[1],location[0]]:continue
                next_enemy_map[location[1],location[0]] += probability*self.enemy_penalty
                walls = self.wall_map[location[1]][location[0]]
                for possible_direction,not_valid in enumerate(walls):
                    if not_valid:
                        continue
                    adjx,adjy= self.change_direction(location[0],location[1],possible_direction,1,0)
                    if not self.validate_location((adjx,adjy)):
                        continue
                    bfsqueuenext.append((adjx,adjy))
            bfsqueuenext,bfsqueue= bfsqueue,bfsqueuenext
            probability *= self.enemy_gamma
            if probability<0.1:
                break
            number +=1
            # if number >3:break
        self.current_enemy_map+=next_enemy_map


        

        
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
    def manhattan_distance(self,location1,location2):
        return abs(location1[0]-location2[0]) + abs(location1[1]-location2[1])
    def add_last_known_position(self,location):
        # if self.is_scout:
        #     print(location)
        self.last_known_location.append(location)
        
        # for index,last_known in enumerate(self.last_known_location[::-1]):
        #     positions.append(index)
        #     if self.manhattan_distance(last_known,location)<=3:
        #         self.last_known_location.pop(index)
        #         self.last_known_location.append(last_known)
        #         return
        while len(self.last_known_location) > 3:
            self.last_known_location.pop(0)
        self.last_known_location.append(location)
    def get_bit(self,integer,position):
        return (integer>>position)&1
    def get_action_scout(self,observation):
        viewcone = observation["viewcone"]
        # viewcone = np.rot90(viewcone, k=-1)
        viewcone = np.rot90(viewcone, k=-1)
        viewcone = np.fliplr(viewcone)

        self.location = observation["location"]
        direction = observation["direction"]
        self.update_map(observation)
        
        # print("walls",self.wall_map[:6,:6])4

        self.calculate_value_map()
        self.calculate_guards_danger()
        
        self.final_value = np.add(np.add(self.accumulated_value_map,self.current_enemy_map),self.permanent_value)
        location = self.location
        current_value = -99999
        action_to_take = 0 
        for direction in range(4):
            next_x,next_y = self.change_direction(location[0],location[1],direction,1,0)
            
            if not self.validate_location((next_x,next_y)):
                continue
            if direction == observation["direction"]:
                self.final_value[next_y,next_x] +=1
            # print(self.final_value[next_y,next_x],end=" ")
            if self.wall_map[self.location[1],self.location[0]][direction] == 0 and self.final_value[next_y,next_x]>current_value:
                current_value = self.final_value[next_y,next_x]
                action_to_take = direction
        # print(self.wall_map[location[1],location[0]],action_to_take)
        # print()
        if self.old_location.count(tuple(self.location))>3:
            return np.random.randint(0,4)
        self.old_location.append(tuple(self.location))
        while len(self.old_location)>5:
            self.old_location.pop(0)
        direction = observation["direction"]
        if action_to_take == direction:
            return 0
        if action_to_take == (direction+2)%4:
            return 1
        if (action_to_take+1)%4 == direction:
            return 2
        if (action_to_take-1)%4 == direction:
            return 3
    def guard_stay_away(self,location):
        current_distance_map = np.full(shape=(self.size,self.size),fill_value=0)
        distance_queue = deque()
        distance_queue.append((0,tuple(location)))
        base_distance = 5
        calls = 0
        while distance_queue:
            calls +=1
            
            current_info = distance_queue.popleft()
            # print(current_info)
            distance,location = current_info
            if distance<1:
                break
            if current_distance_map[location[1]][location[0]]!=0:
                continue
            current_distance_map[location[1],location[0]]=distance
            walls = self.wall_map[location[1]][location[0]]
            for possible_direction,valid in enumerate(walls):
                if valid:
                    continue
                check_location = self.change_direction(location[0],location[1],possible_direction,1,0)
                if not self.validate_location(check_location):
                    continue
                if current_distance_map[check_location[1]][check_location[0]]==-1:
                    distance_queue.append((distance-1,check_location))
        self.target_distance_map = np.add(self.target_distance_map,current_distance_map)
    def calculate_target(self,target):
        self.target_distance_map = np.full(shape=(self.size,self.size),fill_value=-1)
        distance_queue = deque()
        distance_queue.append((0,tuple(target)))
        base_distance = 250
        calls = 0
        while distance_queue:
            calls +=1
            
            current_info = distance_queue.popleft()
            # print(current_info)
            distance,location = current_info
            if self.target_distance_map[location[1]][location[0]]!=-1:
                continue
            self.target_distance_map[location[1],location[0]]=distance
            walls = self.wall_map[location[1]][location[0]]
            for possible_direction,valid in enumerate(walls):
                if valid:
                    continue
                check_location = self.change_direction(location[0],location[1],possible_direction,1,0)
                if not self.validate_location(check_location):
                    continue
                if self.target_distance_map[check_location[1]][check_location[0]]==-1:
                    distance_queue.append((distance+1,check_location))
        # print(self.target_distance_map)
    def update_map(self,observation):
        viewcone = observation["viewcone"]
        viewcone = np.rot90(viewcone, k=-1)
        viewcone = np.fliplr(viewcone)
        self.location = observation["location"]
        self.permanent_value[self.location[1],self.location[0]] -=2
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
                    if has_wall:
                        otherside_x,otherside_y = self.change_direction(currentx,currenty, wall_direction,1,0)
                        if self.validate_location((otherside_x,otherside_y)):
                            self.wall_map[otherside_y][otherside_x][(wall_direction+2)%4] =has_wall
                # print("wallmap",np.binary_repr(info, width=8),self.wall_map[currenty][currentx],direction)
                # if currentx == x and currenty == y:
                    # print(self.wall_map[currenty][currentx])
        dot_array = np.array([16,32,64,128],dtype=np.uint8)
        value_map_array = np.array([16,32,64,128],dtype=np.uint8)
        self.combined_map = np.tensordot(self.wall_map,dot_array,axes=((2,0)))
        self.combined_map = np.uint8(self.combined_map)
        # print(self.combined_map)
    def update_map_guard(self,observation):
        viewcone = observation["viewcone"]
        viewcone = np.rot90(viewcone, k=-1)
        viewcone = np.fliplr(viewcone)
        self.location = observation["location"]

        x,y = self.location#.tolist()
        x,y= int(x),int(y)
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
                if self.get_bit(info,2):
                    self.current_enemy_seen = True
                    self.add_last_known_position((currentx,currenty))
                else:
                    self.current_enemy_seen = False
                if self.get_bit(info,3):
                    self.last_seen_guard.append((currentx,currenty))
                    # self.add_last_known_position((currentx,currenty))
    def get_action_guard(self,observation):
        if self.reverse:
            self.reverse = False
            return 1
        viewcone = observation["viewcone"]
        viewcone = np.rot90(viewcone, k=-1)
        viewcone = np.fliplr(viewcone)  
        self.location = observation["location"]
        # print(self.location)
        x,y = self.location#.tolist()
        direction = observation["direction"]
        self.last_seen_guard = []
        self.update_map(observation)
        self.update_map_guard(observation)
        # while len(self.last_known_location)>1 and (abs(x-self.last_known_location[-1][0])<=2 or abs(y-self.last_known_location[-1][1])<=2):
        #     self.last_known_location.pop(0)
        while len(self.last_known_location)>1:
            self.last_known_location.pop(0)
        if self.last_known_location and self.manhattan_distance(self.last_known_location[0],self.location)<=1:
            location = self.last_known_location[0]
            smallest_distance = 255
            self.chosen_possible = 0
            
            for i in range(4):
                if self.manhattan_distance(location,self.possible_location[i])<smallest_distance:
                    smallest_distance = self.manhattan_distance(location,self.possible_location[i])
                    self.chosen_possible = i
            
            self.last_known_location.pop()
        self.chosen_possible = -1
        if self.last_known_location:
            self.calculate_target(self.last_known_location[-1])
        
        elif self.chosen_possible !=-1 and self.manhattan_distance(self.location,self.possible_location[ self.chosen_possible])<=2:
            self.chosen_possible = -1
        elif self.chosen_possible != -1:
            self.calculate_target(self.possible_location[self.chosen_possible])
        else:
            if self.manhattan_distance(self.location,self.loc_target[self.loc_target_index])<=1:
                self.loc_target_index = (self.loc_target_index+1)%3
            # if len(self.loc_target) == 0:
            #     self.loc_target.append((np.random.randint(0,15),np.random.randint(0,15)))
            print("enemy target is ",self.loc_target[self.loc_target_index])
            self.calculate_target(self.loc_target[self.loc_target_index])
            
        for location in self.last_seen_guard:
            self.guard_stay_away(location)
        while len(self.last_known_location)>1:
            self.last_known_location.pop(0)
        location = self.location
        current_value = 99999
        action_to_take = 0 
        for direction in range(4):
            next_x,next_y = self.change_direction(location[0],location[1],direction,1,0)
            if not self.validate_location((next_x,next_y)):
                continue
            if direction == observation["direction"]:
                self.target_distance_map[next_y,next_x] -=1
            # print(self.final_value[next_y,next_x],end=" ")
            if self.wall_map[self.location[1],self.location[0]][direction] == 0 and self.target_distance_map[next_y,next_x]<current_value:
                current_value = self.target_distance_map[next_y,next_x]
                action_to_take = direction
        # print(self.location,self.wall_map[location[1],location[0]],action_to_take)
        # print()
        direction = observation["direction"]
        if action_to_take == direction:
            return 0
        if action_to_take == (direction+2)%4:
            self.reverse = True
            return 1
        if (action_to_take+1)%4 == direction:
            return 2
        if (action_to_take-1)%4 == direction:
            return 3
    def get_action(self,obs):
        if type(obs) == np.ndarray:
            obs = np.uint8(obs.flatten())
            if len(obs)%289 == 0:
                obs = obs[-294:]
            else:
                legal_actions = obs[-5:]
                obs = obs[-294:-5]
            # print(obs)
            direction = obs[:4]
            direction = int(np.where(direction == 1)[0])
            location = obs[4:6].tolist()
            scout =obs[6:8]
            scout = np.where(scout ==1)[0]
            steps = int(obs[8])
            viewcone = obs[9:]
            viewcone = np.reshape(viewcone,shape=(7,5,8))
            obs = {
                "viewcone": viewcone,
                
                "direction":direction,
                "location": location,
                "scout": int(scout),
                "step": steps,
            }
        if type(obs) == dict:
            viewcone = obs["viewcone"]
            if type(viewcone) == np.ndarray and viewcone.shape == (7,5,8):
                bit_weights = 1 << np.arange(0, 8, 1)
                viewcone = np.tensordot(viewcone, bit_weights, axes=([2], [0])).astype(np.uint8)
                obs["viewcone"] = viewcone
                # print("swapped")
            
        self.step+=1

        
        if self.is_scout:
            action = self.get_action_scout(obs)
        else:
            action = self.get_action_guard(obs)
        # if self.first  and  self.is_scout:
        #     print(self.is_scout)
        #     # k+=1
        #     self._render_frame()
        return action
    def get_type(obs):
        if type(obs) == np.ndarray:
            
            obs = np.uint8(obs.flatten())
            if len(obs)%289 == 0:
                obs = obs[-294:]
            else:
                obs = obs[-294:-5]
            # print(obs)
            direction = obs[:4]
            direction = int(np.where(direction == 1)[0])
            location = obs[4:6].tolist()
            scout =obs[6:8]
            scout = np.where(scout ==1)[0]
            steps = int(obs[8])
            viewcone = obs[9:]
            viewcone = np.reshape(viewcone,shape=(7,5,8))
            obs = {
                "viewcone": viewcone,
                
                "direction":direction,
                "location": location,
                "scout": int(scout),
                "step": steps,
            }
        if type(obs) == dict:
            viewcone = obs["viewcone"]
            if type(viewcone) == np.ndarray and viewcone.shape == (7,5,8):
                bit_weights = 1 << np.arange(0, 8, 1)
                viewcone = np.tensordot(viewcone, bit_weights, axes=([2], [0])).astype(np.uint8)
                obs["viewcone"] = viewcone
                # print("swapped")
        return obs["scout"]
    def test(self):
        self.env.reset()
        obs = self.env.observe("player_0")
        # obs,reward,terminated,truncated,info = self.env.last()
        # print(obs)
        self.get_action(obs)
        # print(self.accumulated_value_map)
        # print(np.round(self.current_enemy_map,decimals=1))
        # print(np.round(self.final_value,decimals=1))
    
    def _render_frame(self):

        img = np.ones((self.window_size,self.window_width,3))

        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels
        pix_square_size = int(pix_square_size)
        # add gridlines
        # self._draw_gridlines(self.size, self.size, pix_square_size)
        for i in range(16):
            cv.line(img,(i*pix_square_size,0),(i*pix_square_size,768),(128,0,0),1)
            cv.line(img,(768,i*pix_square_size),(0,i*pix_square_size),(128,0,0),1)
        # draw environment tiles
        corners = [[pix_square_size/2,-pix_square_size/2],[pix_square_size/2,pix_square_size/2],[-pix_square_size/2,pix_square_size/2],[-pix_square_size/2,-pix_square_size/2]]
        # print(corners)
        for x, y in np.ndindex((self.size, self.size)):
            tile = self.wall_map[y][x]
            # draw whether the tile contains points
            # Tile(tile % 4).draw(self.window, x, y, pix_square_size)
            #tile is an array of size 4, index 0 is right, 1 is down 2 is left, 3 is up
            centerx,centery = (x)*pix_square_size + pix_square_size/2,(y)*pix_square_size + pix_square_size/2
            centerx,centery = int(centerx),int(centery)
            value = self.value_map[y,x]
            if value in [1,5]:
                # print(x,y,value)
                cv.putText(img,str(value),(centerx-pix_square_size//2,centery-pix_square_size//2),color=(0,0,0),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=0.5)
            if self.is_scout:
                # cv.putText(img,str(round(self.final_value[y,x],1)),(centerx-pix_square_size//4,centery+pix_square_size//2-5),color=(0,0,128),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=0.3)
                cv.putText(img,str(round(self.final_value[y,x],1)),(centerx-pix_square_size//4,centery+pix_square_size//2-5),color=(0,0,128),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=0.3)
            else:
                cv.putText(img,str(round(self.target_distance_map[y,x],1)),(centerx-pix_square_size//4,centery+pix_square_size//2-5),color=(0,0,128),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=0.3)

            # print("start",centerx,centery)
            for direction, is_there in enumerate(tile):
                if not is_there:
                    continue
                x1,y1 = centerx+corners[direction][0],centery+corners[direction][1]
                x2,y2 = centerx+corners[(direction+1)%4][0],centery+corners[(direction+1)%4][1]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                cv.line(img,(x1,y1),(x2,y2),(255,0,0),7)
        # print(x,y)
        x,y = self.location
        centerx,centery = (x)*pix_square_size + pix_square_size/2,(y)*pix_square_size + pix_square_size/2
        centerx,centery = int(centerx),int(centery)
        cv.circle(img,(centerx,centery),radius = 12,color=(0,128,0),thickness=-1)
        for positions in self.last_known_location:
            x,y = positions
            centerx,centery = (x)*pix_square_size + pix_square_size/2,(y)*pix_square_size + pix_square_size/2
            centerx,centery = int(centerx),int(centery)
            cv.circle(img,(centerx,centery),radius = 12,color=(0,0,128),thickness=-1)
        cv.imshow("Test",img) 

if __name__ == "__main__":
    # temp = Scout()
    # test_obs = {'viewcone': [[0, 0, 0, 0, 0], [66, 35, 0, 0, 0], [3, 2, 74, 66, 67], [2, 2, 3, 2, 2], [2, 2, 2, 2, 2], [18, 2, 2, 2, 2], [0, 130, 18, 18, 2]], 'direction': 0, 'location': [3, 9], 'scout': 0, 'step': 0}
    # # print(temp.change_direction(0,0,3,0,1))
    # # temp.get_action(test_obs)
    # temp.test()
    print("gay")

    # print(temp.distance_map)
