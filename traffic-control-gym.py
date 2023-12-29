import gym
from gym import spaces
import pygame
import numpy as np
import time
import math
import random


# A Code made by Nathanael Senjaya 2023
# github/NyanNat/Intersection-Reinforcement-AIGym
# A simulation of a real intersection, ported to OpenAI Gym, visualized using PyGame

class cars:
            def __init__(self, source_location, intent_location, screen, car_index):

                # Pre-determined pathing coordinate for each car to follow, based on their source and destination
                self.pathing = {
                    (0,3) : [(450, 450), (480, 300), (450, 150), (440, 0)],
                    (0,5) : [(410, 300),(380, 240), (320, 230), (250, 270), (0, 270)],
                    (0,7) : [(410, 300),(380, 240), (320, 230), (250, 270), (250, 600)],
                    (0,8) : [(480, 435),(525, 400), (665,400), (665, 600)],
                    (0,9) : [(480, 435),(525, 400), (900,400)],
                    (4,5) : [(250, 160), (180, 200), (150, 200), (0, 200)],
                    (6,3) : [(250, 335), (350, 380), (415, 300), (400, 0)],
                    (6,5) : [(250, 335), (350, 380),(415, 300), (330, 205), (260, 265), (0, 275)],
                    (6,7) : [(160, 410), (160, 420), (180, 450), (180, 600)],
                    (6,8) : [(150, 370), (330, 420), (450, 400), (665, 400), (665, 600)],
                    (6,9) : [(150, 370), (330, 420), (450, 370), (900, 370)],
                    (8,3) : [(695, 330), (660, 200), (480, 200), (480, 0)],
                    (8,5) : [(695, 360), (660, 275), (500, 270), (330, 200), (180, 235), (0, 235)],
                    (8,7) : [(695, 360), (660, 275), (500, 270), (380,240), (320, 230), (250,270), (250, 600)],
                    (8,8) : [(695, 435), (670, 400), (670, 600)],
                    (8,9) : [(695, 435), (702, 400), (900, 400)],
                    (10,3) : [(480, 200), (480, 0)],
                    (10,5) : [(500, 235), (330, 200), (180, 235), (0, 235)],
                    (10,7) : [(480, 270), (380,240), (320, 230), (250,270), (250, 600)],
                    (10,8) : [(480, 270), (330,220), (260, 260), (270, 370), (360, 370), (450, 400), (665, 400), (665, 600)],
                    (10,9) : [(480, 270), (330,220), (260, 260), (270, 370), (360, 370), (450, 325), (900, 325)],
                    (11,3) : [(690, 200), (480, 200), (480, 0)],
                    (11,5) : [(690, 200), (330, 200), (180, 235), (0, 235)],
                    (11,7) : [(690, 200), (520, 200), (320, 230), (250,270), (250, 600)],
                    (11,8) : [(690, 180), (665, 400), (665,600)],
                    (11,9) : [(690, 200), (520, 200), (330,220), (260, 260), (270, 370), (360, 370), (450, 325), (900, 325)]
                }

                # Pre-determined spawn location for each car, based on their source and destination
                self.spawn_location = {
                    (0,3) : [440, 604],
                    (0,5) : [408, 604],
                    (0,7) : [408, 604],
                    (0,8) : [480, 604],
                    (0,9) : [480, 604],
                    (4,5) : [250, -4],
                    (6,3) : [-4, 330],
                    (6,5) : [-4, 330],
                    (6,7) : [-4, 410],
                    (6,8) : [-4, 370],
                    (6,9) : [-4, 370],
                    (8,3) : [700,604],
                    (8,5) : [700,604],
                    (8,8) : [700,604],
                    (8,7) : [700,604],
                    (8,9) : [700,604],
                    (10,3) : [904, 200],
                    (10,5) : [904, 230],
                    (10,7) : [904, 270],
                    (10,8) : [904, 270],
                    (10,9) : [904, 270],
                    (11,3) : [700,-4],
                    (11,5) : [700,-4],
                    (11,7) : [700,-4],
                    (11,8) : [700,-4],
                    (11,9) : [700,-4],
                }

                self.other_car_index = []
                self.waiting_time = 0
                self.state = 0
                self.waiting_state = 0
                self.current_time_wait = 0
                self.waiting_time_start = 0

                self.car_index = car_index
                self.velocity = 4                                       # speed in which the vehicle is moving
                self.source_location = source_location                  # Setting the source of car
                self.intent_location = intent_location                  # Setting the destination of car
                self.locationX, self.locationY = self.spawn_location[(int(source_location), int(intent_location))]
                self.move_path = self.pathing[(int(source_location),int(intent_location))]
                self.index = 0
                self.nextLocationX, self.nextLocationY = self.move_path[self.index]
                self.vehicle_size = 7                                   # The size of car (circle)
                self.collision_detection_range = 12                     # The range of semi-circle for other vehicle detection area
                self.window = screen
                self.coll_angle_min, self.coll_angle_max = self.angle_collision_detection(self.dist_x_count(), self.dist_y_count())

            # Counting the angle for collision detection
            def angle_collision_detection(self, x_coor_diff, y_coor_diff):
                if y_coor_diff == 0:
                    if self.dist_x_count() > 0:
                        rad_angle = 0
                    elif self.dist_x_count() < 0:
                        rad_angle = 3.1415
                    else:
                        rad_angle = 5.5
                elif x_coor_diff == 0:
                    if self.dist_y_count() > 0:
                        rad_angle = 4.71239
                    elif self.dist_y_count() < 0:
                        rad_angle = 1.5708
                    else:
                        rad_angle = 5.5
                else:
                    rad_angle = math.atan2(y_coor_diff, x_coor_diff)

                rad_angle_min = rad_angle-0.55
                rad_angle_max = rad_angle + 0.55
                rad_angle_min = math.fmod(rad_angle_min, 2 * math.pi)
                rad_angle_max = math.fmod(rad_angle_max, 2 * math.pi)
                return rad_angle_min, rad_angle_max

            # For checking whether a car or traffic light is near other car to collide
            def collision_detection(self, own_x, own_y, other_x, other_y):
                rad_angle = 3.1415

                x_diff = other_x - own_x
                y_diff = other_y - own_y
                distance = self.count_distance(x_diff, y_diff)

                if y_diff == 0:
                    if x_diff > 0:
                        rad_angle = 0
                    elif x_diff < 0:
                        rad_angle = 3.1415
                elif x_diff == 0:
                    if y_diff > 0:
                        rad_angle = 4.71239
                    elif y_diff < 0:
                        rad_angle = 1.5708
                else:
                    rad_angle = math.atan2(y_diff, x_diff)

                if rad_angle > self.coll_angle_min and rad_angle < self.coll_angle_max and distance < self.collision_detection_range:
                    return True
                else:
                    return False

            def count_distance(self, x, y):
                return ((x) ** 2 + (y) ** 2) ** 0.5

            def dist_x_count(self):
                return self.nextLocationX - self.locationX

            def dist_y_count(self):
                return self.nextLocationY - self.locationY

            def move(self, traffic_light_location, traffic_light_state):
                distance = self.count_distance(self.dist_x_count(), self.dist_y_count())
                for _, car_loc in self.other_car_index:
                    if self.collision_detection(self.locationX, self.locationY, car_loc[0], car_loc[1]):
                        if self.waiting_state == 0:
                            self.waiting_time_start = time.time()
                            self.current_time_wait = time.time()
                            self.waiting_state = 1
                        elif self.waiting_state == 1:
                            self.current_time_wait = time.time()
                        self.waiting_time = self.current_time_wait - self.waiting_time_start
                        return
                    
                for i, traffic_light in enumerate(traffic_light_location):
                    if traffic_light_state[i] != 1:
                        for loc_light in traffic_light:
                            if self.collision_detection(self.locationX, self.locationY, loc_light[0], loc_light[1]):
                                if self.waiting_state == 0:
                                    self.waiting_time_start = time.time()
                                    self.current_time_wait = time.time()
                                    self.waiting_state = 1
                                elif self.waiting_state == 1:
                                    self.current_time_wait = time.time()
                                self.waiting_time = self.current_time_wait - self.waiting_time_start
                                return

                if distance < self.velocity:
                    if self.waiting_state == 1:
                        self.waiting_time = 0
                        self.waiting_state = 0
                    self.index = self.index+1
                    self.locationX, self.locationY = self.nextLocationX, self.nextLocationY
                    if self.index == len(self.move_path):
                        self.nextLocationX, self.nextLocationY = self.locationX, self.locationY
                        self.state = 1
                        return
                    elif self.index < len(self.move_path):
                        self.nextLocationX, self.nextLocationY = self.move_path[self.index]
                        self.coll_angle_min, self.coll_angle_max = self.angle_collision_detection(self.dist_x_count(), self.dist_y_count())
                        return
                else:
                    if self.waiting_state == 1:
                        self.waiting_time = 0
                        self.waiting_state = 0
                    ratio = self.velocity / distance
                    self.locationX += self.dist_x_count() * ratio
                    self.locationY += self.dist_y_count() * ratio

            def draw(self):
                pygame.draw.circle(self.window,(0, 0, 255), (self.locationX, self.locationY), self.vehicle_size)

            # To give information about the current location coordinate of the car
            def get_location(self):
                return (self.locationX, self.locationY)

            # To give information about the car_index (the number assigned when car is spawned)
            def get_car_index(self):
                return self.car_index

            # To add the knowledge of previously established car during instantiation
            def add_other_car_index(self, other_car_loc):
                self.other_car_index = other_car_loc[:]

            # To add new data to other car index on the right side
            def append_other_car_index(self, data):
                self.other_car_index.append(data)

            # To update one certain index coordinate in other_car_index array
            def update_one_index_other_car(self, index_value, data):
                first_column = [row[0] for row in self.other_car_index]
                try:
                    arr_index = first_column.index(index_value)
                    self.other_car_index[arr_index][1] = data
                except:
                    return

            # To remove a certain index coordinate from other_car_index array
            def remove_other_car_loc(self, index_value):
                first_column = [row[0] for row in self.other_car_index]
                arr_index = first_column.index(index_value)
                self.other_car_index.pop(arr_index)

            def check_waiting(self):
                return self.waiting_time


class TrafficControlEnv(gym.Env):
    def __init__(self, seed = 0, END_TIME = 120):

        self.window_height = 600    # The height of the PyGame window
        self.window_width = 900     # the width of the pygame window

        self.start_env_time = time.time()
        self.END_TIME = END_TIME

        self.active_cars_array = []                     # Array of cars object that are currently on road
        self.cars_reached = []                          # Arrays of cars object that has reached its final checkpoint at that current time
        self.active_cars_locations = []                 # Arrays of all cars object that are currently in the road

        # Spawn time for each lane based on random and the seed
        self.spawn_time =  [random.uniform(3, 3.5), random.uniform(4, 4.5), random.uniform(3, 3.5), random.uniform(4, 4.5), random.uniform(3.5, 4), random.uniform(4, 4.5)]
        # Array to store the running time of each lane
        self.init_time = [time.time(), time.time(), time.time(), time.time(), time.time(), time.time()]
        self.eligible_source = [0, 4, 6, 8, 10, 11]     # List of all eligible source of cars moving into the intersection
        self.eligible_destination = [3, 5, 7, 8, 9]     # List of all eligible destination of cars leaving from the intersection
        self.eligible_time = [10, 15, 20]               # List of all eligible time for the system to choose

        self.traffic_color = {
            0 : (255, 0, 0),        # red
            1 : (0, 255, 0)         # green
        }

        self.traffic_light_location = [[(x, y) for x in range(390, 501) for y in range(442,443)], [(x, y) for x in range(518,519) for y in range(180,291)],
                                        [(x, y) for x in range(160, 271) for y in range(158,159)], [(x, y) for x in range(143,144) for y in range(310,421)],
                                        [(x, y) for x in range(680, 716) for y in range(423,424)], [(x, y) for x in range(720,721) for y in range(180,290)],
                                        [(x, y) for x in range(680, 716) for y in range(179,180)], [(x, y) for x in range(638,639) for y in range(310,421)],
                                        [(x, y) for x in range(332, 333) for y in range(330,370)], [(x, y) for x in range(328, 329) for y in range(210,240)]]
        #kucing
        self.traffic_light_state = np.zeros(10, dtype=np.int32)          # Array of states of each traffic light
        self.car_num_each_intersection = np.zeros(10, np.int32)
        self.total_waiting = np.zeros(10, dtype=np.float32)

        self.cars_spawned = 0
        self.car_passed = 0

        # Applying a seed for the traffic light env
        random.seed(seed)

        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()

        #observation space represent each traffic lights that can be changed in each intersection
        self.observation_space = spaces.Tuple((spaces.MultiBinary(10), spaces.Box(low=0, high=np.inf, shape=(10,), dtype=np.float32),
                                               spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)))

        self.action_space = spaces.Tuple((spaces.MultiBinary(10), spaces.Discrete(3)))

        self.reset()

    # Update the positions of each active vehicles based on after their movement
    def update_vehicles(self, traffic_light_location):
        for car in self.active_cars_array:
            car.move(traffic_light_location, self.traffic_light_state)
            index = car.get_car_index()
            data = car.get_location()
            for car in self.active_cars_array:
                car.update_one_index_other_car(index, data)
        self.update_all_other_vehicle()
    
    # Update positions of each active vehicle
    def update_all_other_vehicle(self):
        for car in self.active_cars_array:
            index = car.get_car_index()
            data = car.get_location()
            if index == 0:
                return
            else:
                for car in self.active_cars_array:
                    car.update_one_index_other_car(index, data)
    
    def check_number_each_light(self):
        self.car_num_each_intersection = np.zeros(10, dtype=np.int32)
        self.total_waiting = np.zeros(10, dtype=np.float32)

        for i, loc in enumerate(self.active_cars_locations):
            if loc[1][0] > 390 and loc[1][0] < 500 and loc[1][1] >= 440 and self.traffic_light_state[0] != 1:
                self.car_num_each_intersection[0] += 1
                self.total_waiting[0] += self.active_cars_array[i].check_waiting()
            elif loc[1][1] > 180 and loc[1][1] < 290 and loc[1][0] >= 515 and loc[1][0] <= 715 and self.traffic_light_state[1] != 1:
                self.car_num_each_intersection[1] += 1
                self.total_waiting[1] += self.active_cars_array[i].check_waiting()
            elif loc[1][0] > 160 and loc[1][0] < 270 and loc[1][1] <= 160 and self.traffic_light_state[2] != 1:
                self.car_num_each_intersection[2] += 1
                self.total_waiting[2] += self.active_cars_array[i].check_waiting()
            elif loc[1][1] > 310 and loc[1][1] < 420 and loc[1][0] <= 140 and self.traffic_light_state[3] != 1:
                self.car_num_each_intersection[3] += 1
                self.total_waiting[3] += self.active_cars_array[i].check_waiting()
            elif loc[1][0] > 680 and loc[1][0] < 715 and loc[1][1]>= 420 and self.traffic_light_state[4] != 1:
                self.car_num_each_intersection[4] += 1
                self.total_waiting[4] += self.active_cars_array[i].check_waiting()
            elif loc[1][1] > 180 and loc[1][1] < 290 and loc[1][0] >= 718 and self.traffic_light_state[5] != 1:
                self.car_num_each_intersection[5] += 1
                self.total_waiting[5] += self.active_cars_array[i].check_waiting()
            elif loc[1][1] > 680 and loc[1][1] < 715 and loc[1][0] <= 176 and self.traffic_light_state[6] != 1:
                self.car_num_each_intersection[6] += 1
                self.total_waiting[6] += self.active_cars_array[i].check_waiting()
            elif loc[1][1] > 310 and loc[1][1] < 420 and loc[1][0] >= 515 and loc[1][0] <= 638 and self.traffic_light_state[7] != 1:
                self.car_num_each_intersection[7] += 1
                self.total_waiting[7] += self.active_cars_array[i].check_waiting()
            elif loc[1][1] > 300 and loc[1][1] < 390 and loc[1][0] >= 250 and loc[1][0] <= 330 and self.traffic_light_state[8] != 1:
                self.car_num_each_intersection[8] += 1
                self.total_waiting[8] += self.active_cars_array[i].check_waiting()
            elif loc[1][1] > 210 and loc[1][1] < 300 and loc[1][0] >= 325 and loc[1][0] <= 410 and self.traffic_light_state[9] != 1:
                self.car_num_each_intersection[9] += 1
                self.total_waiting[9] += self.active_cars_array[i].check_waiting()

        for i, tot_wait in enumerate(self.total_waiting):
                try:
                    self.total_waiting[i] = tot_wait*0.001
                except ZeroDivisionError:
                    self.total_waiting[i] = 0
        return self.car_num_each_intersection, self.total_waiting

    # To remove a certain car object for every currently active cars
    def update_location_array(self):
        if len(self.cars_reached_index) > 0:
            for index in self.cars_reached_index:
                for car in self.active_cars_array:
                        if car.get_car_index() != index:
                            car.remove_other_car_loc(index)
                first_column = [row[0] for row in self.active_cars_locations]
                arr_index = first_column.index(index)
                self.active_cars_locations.pop(arr_index)

        self.update_all_other_vehicle()
    
    def spawn_car(self, source):
            if source == 4:
                to = 5
            else:
                to = random.choice(self.eligible_destination)

            # Instatiate a car object with index number according to self.cars_spawned
            car = cars(source, to, self.window, self.cars_spawned)

            #print(f'\nCar {self.cars_spawned} is spawned with coordinate {car.get_location()}\n\n')                    # For debugging purposes

            # Appending the index number of a car and its location of the newly spawned car to every car that are already active
            for carObject in self.active_cars_array:
                carObject.append_other_car_index([self.cars_spawned, car.get_location()])
            
            # Giving the newly spawned car the knowledge of all already active cars
            car.add_other_car_index(self.active_cars_locations)

            # Appending new car object to the active_cars_array
            self.active_cars_array.append(car)
            
            #Appending new car object index number and its location into active_cars_locations
            self.active_cars_locations.append([self.cars_spawned, car.get_location()])

            self.update_all_other_vehicle()
            self.cars_spawned += 1

    # To draw vehicle to canvas
    def draw_vehicles(self):
        for car in self.active_cars_array:
            car.draw()

    def reset(self):
        # Changing all the current traffic light to red
        self.traffic_light_state = np.zeros(10, dtype=np.int32)
        self.car_num_each_intersection = np.zeros(10, dtype=np.int32)
        self.total_waiting = np.zeros(10, dtype=np.float32)
        self.car_passed = 0

        # Remove all cars from frame
        self.active_cars_array = []
        self.active_cars_locations = []

        # Render the reseted frame in pygame
        #self._render_frame()
        return self.car_num_each_intersection, self.total_waiting, [0] 


    def step(self, action_step):
        reward = 0
        terminated = False

        light_step = action_step[0].reshape(-1)

        # Adding the time index for each traffic lane that is turned green
        for i, act in enumerate(light_step):
            if act == 1:
                reward += self.total_waiting[i]
        
        self.traffic_light_state = light_step#np.array(action_step[0], dtype=int)
        waiting_time = self.eligible_time[action_step[1]]

        # Start timer per requested waiting time
        start_time_wait = time.time()

        while time.time() - start_time_wait <= waiting_time:
            return_obs_time = time.time() - self.start_env_time
            if return_obs_time >= self.END_TIME:
                terminated = True
                break

            # Spawn car
            for i, source in enumerate(self.eligible_source):
                current_time = time.time()
                if current_time - self.init_time[i] >= self.spawn_time[i]:
                    self.init_time[i] = current_time
                    self.spawn_car(source)
            
            # Update the positions of vehicles
            self.update_vehicles(self.traffic_light_location)

            # remove cars that has reached destination
            self.cars_reached = [car for car in self.active_cars_array if car.state == 1]
            self.active_cars_array = [car for car in self.active_cars_array if car.state == 0]

            # List the index of the cars that have reached destination
            self.cars_reached_index = [car.get_car_index() for  car in self.cars_reached]

            # Reward added with number of cars reaches their own destination
            reward += len(self.cars_reached_index)*0.25
            self.car_passed += len(self.cars_reached_index)

            # To update the lcoation array which contain other cars when a car has reached destination
            self.update_location_array()

            # To get the current update for each coordinate for each still active car and updating the array
            self.update_all_other_vehicle()

            #self._render_frame()


        return_time = self.check_number_each_light()

        '''
        observation = {'cars_num': return_time[0],
                        'waiting_index': return_time[1],
                        'time': return_obs_time}
        '''
        observation = (return_time[0], return_time[1], [return_obs_time])

        if terminated:
            self.start_env_time = time.time()
        return observation, reward, terminated, {}, self.car_passed

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Traffic Intersection")
        if self.clock is None:
            clock = pygame.time.Clock()

        #update the background of the file
        self.window.fill((255,255,255))
        pygame.draw.rect(self.window, (0, 0, 0), [160, 0, 110, 250], 0)     #4
        pygame.draw.rect(self.window, (0, 0, 0), [390, 0, 110, 250], 0)     #3
        pygame.draw.rect(self.window, (0, 0, 0), [160, 350, 110, 250], 0)   #7
        pygame.draw.rect(self.window, (0, 0, 0), [390, 350, 110, 250], 0)   #0
        pygame.draw.rect(self.window, (0, 0, 0), [350, 180, 550, 110], 0)   #2, 10
        pygame.draw.rect(self.window, (0, 0, 0), [350, 310, 550, 110], 0)   #1, 9
        pygame.draw.rect(self.window, (0, 0, 0), [0, 180, 250, 110], 0)     #5
        pygame.draw.rect(self.window, (0, 0, 0), [0, 310, 250, 110], 0)     #6
        pygame.draw.rect(self.window, (0, 0, 0), [680, 0, 35, 600], 0)      #8, 11
        pygame.draw.rect(self.window, (0, 0, 0), [642.5, 400, 37.5, 200], 0)
        pygame.draw.rect(self.window, (0, 0, 0), [650, 250, 100, 100], 0)   #
        pygame.draw.circle(self.window, (0, 0, 0), [330,300], 220)          #
        pygame.draw.circle(self.window, (255, 255, 255), [330,300], 60)     #

        #road border
        pygame.draw.rect(self.window, (255, 255, 0), [425, 440, 2.5, 200], 0)   #0
        pygame.draw.rect(self.window, (255, 255, 0), [462.5, 440, 2.5, 200], 0)   #0
        pygame.draw.rect(self.window, (255, 255, 0), [195, 440, 2.5, 200], 0)   #7
        pygame.draw.rect(self.window, (255, 255, 0), [232.5, 440, 2.5, 200], 0)   #7
        pygame.draw.rect(self.window, (255, 255, 0), [425, 0, 2.5, 160], 0)   #3
        pygame.draw.rect(self.window, (255, 255, 0), [462.5, 0, 2.5, 160], 0)  #3
        pygame.draw.rect(self.window, (255, 255, 0), [195, 0, 2.5, 160], 0)   #4
        pygame.draw.rect(self.window, (255, 255, 0), [232.5, 0, 2.5, 160], 0) #4
        pygame.draw.rect(self.window, (255, 255, 0), [0, 215, 140, 2.5], 0)   #5
        pygame.draw.rect(self.window, (255, 255, 0), [0, 252.5, 140, 2.5], 0)   #5
        pygame.draw.rect(self.window, (255, 255, 0), [0, 345, 140, 2.5], 0)   #6
        pygame.draw.rect(self.window, (255, 255, 0), [0, 382.5, 140, 2.5], 0)   #6
        pygame.draw.rect(self.window, (255, 255, 0), [520, 215, 150, 2.5], 0)   #2
        pygame.draw.rect(self.window, (255, 255, 0), [520, 252.5, 150, 2.5], 0)   #2
        pygame.draw.rect(self.window, (255, 255, 0), [520, 345, 150, 2.5], 0)   #1
        pygame.draw.rect(self.window, (255, 255, 0), [520, 382.5, 150, 2.5], 0)   #1
        pygame.draw.rect(self.window, (255, 255, 0), [730, 215, 180, 2.5], 0)   #10
        pygame.draw.rect(self.window, (255, 255, 0), [730, 252.5, 180, 2.5], 0)   #10
        pygame.draw.rect(self.window, (255, 255, 0), [678.5, 420, 2.5, 180], 0)   #8
        pygame.draw.rect(self.window, (255, 255, 0), [730, 345, 180, 2.5], 0)   #9
        pygame.draw.rect(self.window, (255, 255, 0), [730, 382.5, 180, 2.5], 0)   #9

        #grasses
        pygame.draw.rect(self.window, (255, 255, 255), [270, 475, 120, 250], 0)   #
        pygame.draw.rect(self.window, (255, 255, 255), [270, 0, 120, 130], 0)   #

        # traffic light
        pygame.draw.rect(self.window, self.traffic_color[self.traffic_light_state[0]], [390, 440, 110, 5])      # 0
        pygame.draw.rect(self.window, self.traffic_color[self.traffic_light_state[1]], [515, 180, 5, 110])      # 2
        pygame.draw.rect(self.window, self.traffic_color[self.traffic_light_state[2]], [160, 155, 110, 5])      # 4
        pygame.draw.rect(self.window, self.traffic_color[self.traffic_light_state[3]], [140, 310, 5, 110])      # 6
        pygame.draw.rect(self.window, self.traffic_color[self.traffic_light_state[4]], [680, 420, 35, 5])       # 8
        pygame.draw.rect(self.window, self.traffic_color[self.traffic_light_state[5]], [718, 180, 5, 110])      # 10
        pygame.draw.rect(self.window, self.traffic_color[self.traffic_light_state[6]], [680, 176, 35, 5])       # 11
        pygame.draw.rect(self.window, self.traffic_color[self.traffic_light_state[7]], [636, 310, 5, 110], 0)   # 1
        pygame.draw.rect(self.window, self.traffic_color[self.traffic_light_state[8]], [330, 360, 5, 25], 0)    # 12
        pygame.draw.rect(self.window, self.traffic_color[self.traffic_light_state[9]], [325, 215, 5, 25], 0)    # 13

        # Draw the traffic lights and vehicles
        self.draw_vehicles()

        # Update the Pygame display
        pygame.display.flip()

        # Cap the frame rate
        self.clock.tick(60)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
