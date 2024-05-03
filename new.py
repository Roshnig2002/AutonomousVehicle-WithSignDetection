import random
import time
import numpy as np
import math 
import cv2
import gym
from gym import spaces
import glob
import os
import threading
import sys
import math


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import keras.models
from keras.models import load_model

from ultralytics import YOLO
model = YOLO(r"E:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\best (1).pt")

# cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

SECONDS_PER_EPISODE = 15

N_CHANNELS = 3
HEIGHT = 240
WIDTH = 320

SPIN = 10 #angle of random spin

HEIGHT_REQUIRED_PORTION = 0.5 #bottom share, e.g. 0.1 is take lowest 10% of rows
WIDTH_REQUIRED_PORTION = 0.9

# new yolo traffic sign algorithm
CLASS = 0
stop_event = threading.Event()
# Function to change the global variable 'classs'
def change_classs(stop_event):
    global CLASS
    while not stop_event.is_set():
        # time.sleep(14)
        # CLASS = 22
        # print("Changing classs to:", new_value[0])
        new_value = random.choices([0, 1, 2, 3, 4, 5, 17, 28, 30])#, cum_weights = [25, 25, 25, 25])  # Choose a random value
        print("Changing classs to:", new_value[0])
        CLASS = new_value[0]
        time.sleep(random.uniform(5, 10))  # Sleep for a random time interval between 5 to 10 seconds
# Create and start the thread
thread = threading.Thread(target=change_classs, args=(stop_event,))
thread.start()


SHOW_PREVIEW = True

class CarEnv(gym.Env):
    
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = WIDTH
    im_height = HEIGHT
    im_width_rgb = 640
    im_height_rgb = 480
    front_camera = None
    CAMERA_POS_Z = 1.3 
    CAMERA_POS_X = 1.4
    PREFERRED_SPEED = 70 # what it says
    SPEED_THRESHOLD = 3 #defines when we get close to desired speed so we drop the speed
    last_reward = 0
    
    def __init__(self):
        print("init")
        super(CarEnv, self).__init__()
        self.fileCount = 1
        self.action_space = spaces.Discrete(9)  # 9 different brake values

        self.height_from = int(HEIGHT * (1 -HEIGHT_REQUIRED_PORTION))
        self.width_from = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
        self.width_to = self.width_from + int(WIDTH_REQUIRED_PORTION * WIDTH)
        self.new_height = HEIGHT - self.height_from
        self.new_width = self.width_to - self.width_from
        self.image_for_CNN = None
        self.obsDict = {"velocity": None, "sign": None}
        self.actor_list = []
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(3,1), dtype=np.float32)

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.client.load_world('Town04')
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = not self.SHOW_CAM
        self.world.apply_settings(self.settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.cnn_model = load_model(r"E:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\model_saved_from_CNN.h5",compile=False)
        print("Model loaded successfully")
        self.cnn_model.compile()
        if self.SHOW_CAM:
            self.spectator = self.world.get_spectator()
        

        # while True:
        #     self.world.tick()
        #     print("tick is finished")
    
    def cleanup(self):

        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        cv2.destroyAllWindows()
    
    def maintain_speed(self,s):
        if s >= self.PREFERRED_SPEED:
            return 0
        elif s < self.PREFERRED_SPEED - self.SPEED_THRESHOLD:
            return 0.7 
        else:
            return 0.3
            
    def apply_cnn(self,im):
        img = np.float32(im)
        img = img /255
        img = np.expand_dims(img, axis=0)
        cnn_applied = self.cnn_model([img,0],training=False)
        cnn_applied = np.squeeze(cnn_applied)
        # print(f"output of the cnn model{cnn_applied}")
        return  cnn_applied ##[0][0]
    
    def step(self, action):
        trans = self.vehicle.get_transform()
        if self.SHOW_CAM:
            self.spectator.set_transform(carla.Transform(trans.location + carla.Location(z=20),carla.Rotation(yaw =-180, pitch=-90)))
            

        self.step_counter +=1
        brake = action
        # throttle = action[1]
        if brake == 0:
            brake_value = 0.0
        elif brake == 1:
            brake_value = 0.1
        elif brake == 2:
            brake_value = 0.2
        elif brake == 3:
            brake_value = 0.3
        elif brake == 4:
            brake_value = 0.4
        elif brake == 5:
            brake_value = 0.5
        elif brake == 6:
            brake_value = 0.6
        elif brake == 7:
            brake_value = 0.7
        elif brake == 8:
            brake_value = 0.8
        elif brake == 9:
            brake_value = 0.9
        # if brake<=9:
        #     brake_value = 0.1*brake
        else:
            brake_value = 1.0  # Default value if brake value is out of range

        
        # if steer ==0:
        #     steer = - 0.9
        # elif steer ==1:
        #     steer = -0.25
        # elif steer ==2:
        #     steer = -0.1
        # elif steer ==3:
        #     steer = -0.05
        # elif steer ==4:
        #     steer = 0.0 
        # elif steer ==5:
        #     steer = 0.05
        # elif steer ==6:
        #     steer = 0.1
        # elif steer ==7:
        #     steer = 0.25
        # elif steer ==8:
        #     steer = 0.9
        # if throttle == 0:
        #     throttle = 0.0
        # elif throttle == 1:
        #     throttle = 0.3
        # elif throttle == 2:
        #     throttle = 0.6
            
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        estimated_throttle = self.maintain_speed(kmh)
        if brake_value > 0:
            estimated_throttle = 0
        # print('Estimated throttle:',estimated_throttle)
        self.vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=0.0, brake = brake_value))


        distance_travelled = self.initial_location.distance(self.vehicle.get_location())

        cam = self.front_camera
        rgb_frame = self.front_camera_rgb

        # image = 255 * np.ones((100, 200, 3), dtype=np.uint8)
        # cv2.rectangle(image, (0, 0), (200, 100), (0, 0, 0), -1)

        if self.SHOW_CAM:
            cv2.imshow('Frame', cam)
            cv2.imshow('Frame_RGB', rgb_frame)
            
            # cv2.putText(image, 'Speed: '+str(int(kmh))+' kmh', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.imshow("Velocity Display", image)
            self.world.debug.draw_string(self.vehicle.get_transform().location, f"Velocity: {kmh:.2f} km/h", draw_shadow=False, color=carla.Color(r=255, g=255, b=255), life_time=0.1, persistent_lines=True)

            cv2.waitKey(1)
        # lock_duration = 0
        # if self.steering_lock == False:
        #     if steer<-0.6 or steer>0.6:
        #         self.steering_lock = True
        #         self.steering_lock_start = time.time() 
        # else:
        #     if steer<-0.6 or steer>0.6:
        #         lock_duration = time.time() - self.steering_lock_start
        
        reward = 0
        done = False
        if len(self.collision_hist) != 0:
            done = True
            reward = reward - 1
            self.cleanup()
        # if len(self.lane_invade_hist) != 0:
        #     done = True
        #     reward = reward - 300
        #     self.cleanup()
        # if lock_duration>3:
        #     reward = reward - 150
        #     done = True
        #     self.cleanup()
        # elif lock_duration > 1:
        #     reward = reward - 20


        #reward for acceleration
        # if kmh < 10 and CLASS not in [3, 7, 22]:
        #     reward = reward - 0.5
        # elif kmh <15 and CLASS not in [3, 7, 22]:
        #     reward = reward - 0.5
        # elif kmh>30 and CLASS==0:
        #     reward = reward - 1 #punish for going greater than 30km per hour
        # elif kmh>60 and CLASS==1:
        #     reward = reward - 1 #punish for going greater than 60km per hour
        # elif kmh > 10 and CLASS in [3, 7, 22]:
        #     reward = reward - 1
        # elif kmh > 5 and CLASS in [7, 22]:
        #     reward = reward - 0.7
        # # elif kmh > 0 and CLASS in [3, 7, 22]:
        # #     reward = reward - 2
        # else:
        #     reward = reward + 1


        # Reward Calculation
        if CLASS == 0:  #speed limit 30
            if kmh<30-self.SPEED_THRESHOLD and brake_value>0:
                reward = -1
            elif kmh>30:
                # reward = -abs(brake_value-((1/(self.PREFERRED_SPEED - 30))*kmh))
                reward = -1
            else:
                reward = (1/30)*kmh * 2
        elif CLASS == 1:    #speed limit 60
            if kmh < 60-self.SPEED_THRESHOLD and brake_value>0:
                reward = -1
            elif kmh > 60:
                # reward = -abs(brake_value-((1/(self.PREFERRED_SPEED - 60))*kmh))
                reward = -1
            else:
                reward = (1/60)*kmh*2
        elif CLASS == 2:    #Yield sign
            if kmh<20-self.SPEED_THRESHOLD and brake_value>0:
                reward = -1
            elif kmh>20:
                reward = -1
            else:
                reward = (1/20)*kmh*2
        elif CLASS == 3:    #STOP sign
            if kmh > 0:
                # tempCalc = -0.3 - ((1/70)*kmh)
                # if tempCalc > -1:
                #     reward = tempCalc
                # else:
                #     reward = math.ceil(-0.3 + ((1/70)*kmh))
                reward = -1
            else:
                reward = 2
        elif CLASS == 4:    #Beware of Children sign
            if kmh<25-self.SPEED_THRESHOLD and brake_value>0:
                reward = -1
            elif kmh>25:
                reward = -1
            else:
                reward = (1/25)*kmh*2
        elif CLASS == 5:    #Men at work sign
            if kmh<35-self.SPEED_THRESHOLD and brake_value>0:
                reward = -1
            elif kmh>35:
                reward = -1
            else:
                reward = (1/35)*kmh*2
        elif CLASS == 17:    #Pedestrian sign
            if kmh<30-self.SPEED_THRESHOLD and brake_value>0:
                reward = -1
            elif kmh>30:
                reward = -1
            else:
                reward = (1/30)*kmh*2
        elif CLASS == 28:    #Uneven road sign
            if kmh<20-self.SPEED_THRESHOLD and brake_value>0:
                reward = -1
            elif kmh>20:
                reward = -1
            else:
                reward = (1/20)*kmh*2
        
        elif CLASS == 30:   #No traffic sign
            if brake_value > 0:
                reward = -1
            else:
                reward = 2

        # print(f"reward after calculating for acceleration is {reward} with velocity as {kmh}")

        ## reward for braking

        # expected_dict = { 0 : 0.6, 1 : 0.4, 3 : 0.8, 7 : 0.8, 22 : 0.9, 30 : 0}

        # #loss calculation
        # loss_val = abs(expected_dict[CLASS] - brake_value)
        # temp = (1 - loss_val)*2 - 1
        # print(f"Current reward as per brake level: {temp}")
        # reward -= (1 - loss_val)*2 - 1


        # if brake_value < 0.4 and CLASS==0:  #speed limit 30
        #     reward = reward - 1
        # elif brake_value > 0.7 and CLASS==0:   #speed limit 30
        #     reward = reward - 0.2
        # elif brake_value < 0.3  and CLASS==1:  #speed limit 60
        #     reward = reward - 1 
        # elif brake_value > 0.5  and CLASS==1:  #speed limit 60
        #     reward = reward - 0.2 
        # elif brake_value < 0.7 and CLASS in [3, 7]:  #kids & no entry
        #     reward = reward - 1 
        # elif brake_value < 0.8 and CLASS==22: #stop
        #     reward = reward - 1
        # elif brake_value > 0.0 and CLASS==30: #normal condition (no traffic sign)
        #     reward = reward-1
        # else:
        #     reward = reward + 1
        # reward for making distance
        # if distance_travelled > 1:
        #     reward = reward + 1
        # else:
        #     reward = reward - 0.5


        # elif distance_travelled<50:
        #     reward =  reward + 1
        # else:
        #     reward = reward + 2
        # check for episode duration
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
            # CLASS = 30
            self.cleanup()
        self.image_for_CNN = self.apply_cnn(self.front_camera[self.height_from:,self.width_from:self.width_to])
        # for testing
        toGiveModel = []
        kmhList = []
        kmhList.append(kmh)
        classsList = []
        classsList.append(CLASS)
        currentBreak = []
        currentBreak.append(brake_value)
        toGiveModel.append(kmhList)
        toGiveModel.append(classsList)
        toGiveModel.append(currentBreak)
        # print(toGiveModel)
        # self.obsDict["image_for_CNN"]  = self.image_for_CNN
        self.obsDict["velocity"] = kmh
        self.obsDict["sign"] = CLASS   #YOLO output
        # return self.obsDict, reward, done, {}

        # print("Total reward = ",reward)
        # Simply logging the reward to print later if needed
        self.last_reward = reward

        if self.step_counter % 50 == 0:
            #print('steer input from model:',steer)
            print(f"reward after calculating for acceleration is {reward} with velocity as {kmh}")
            print('Brake input from model:',brake_value, " -- Last Reward: ", reward)

        return toGiveModel, reward, done, {}#self.image_for_CNN, reward, done, self.obsDict    #curly brackets - empty dictionary required by SB3 format

    def reset(self):
        self.collision_hist = []
        # self.lane_invade_hist = []
        self.actor_list = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        
        self.vehicle = None
        while self.vehicle is None:
            try:
        # connect
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            except:
                pass
        self.actor_list.append(self.vehicle)

        # Add RGB camera sensor to the vehicle
        self.rgb_cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam_bp.set_attribute('image_size_x', f"{self.im_width_rgb}")
        self.rgb_cam_bp.set_attribute('image_size_y', f"{self.im_height_rgb}")
        self.rgb_cam_bp.set_attribute('fov', f"110")
        
        # Adjust the location of the camera sensor to capture scenes on the right side
        camera_transform = carla.Transform(carla.Location(x=1.5, y=0.8, z=2.4))  # Adjust location
        self.rgb_cam = self.world.spawn_actor(self.rgb_cam_bp, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_cam)
        self.rgb_cam.listen(lambda data: self.process_img_rgb(data))

        #semantic camera
        self.initial_location = self.vehicle.get_location()
        self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.sem_cam.set_attribute("fov", f"90")
        
        camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA_POS_X))
        self.sensor = self.world.spawn_actor(self.sem_cam, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))
        time.sleep(2)
        
        # now apply random yaw so the RL does not guess to go straight
        # angle_adj = random.randrange(-SPIN, SPIN, 1)
        trans = self.vehicle.get_transform()
        trans.rotation.yaw = trans.rotation.yaw 
        # + angle_adj
        self.vehicle.set_transform(trans)
        
        
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        # self.lanesensor = self.world.spawn_actor(lanesensor, camera_init_trans, attach_to=self.vehicle)
        # self.actor_list.append(self.lanesensor)
        # self.lanesensor.listen(lambda event: self.lane_data(event))
        
        ##adding multiple actors into the environment
         
        # for _ in range(0, 30):
        #     spawn_point = random.choice(self.world.get_map().get_spawn_points())
        #     bp_vehicle = random.choice(self.world.get_blueprint_library().filter('vehicle'))
        #     other_vehicle = self.world.try_spawn_actor(bp_vehicle, spawn_point)
        #     if other_vehicle is not None:
        #         other_vehicle.set_autopilot(True)
        #         self.actor_list.append(other_vehicle) 
  


        while self.front_camera is None:
            time.sleep(0.01)
        
        self.episode_start = time.time()
        # self.steering_lock = False
        # self.steering_lock_start = None # this is to count time in steering lock and start penalising for long time in steering lock
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        self.step_counter = 0
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0))
        self.image_for_CNN = self.apply_cnn(self.front_camera[self.height_from:,self.width_from:self.width_to])
        self.obsDict["image_for_CNN"]  = self.image_for_CNN
        self.obsDict["velocity"] = kmh
        self.obsDict["sign"] = CLASS
        # for testing
        toGiveModel = []
        kmhList = []
        kmhList.append(kmh)
        classsList = []
        classsList.append(CLASS)
        toGiveModel.append(kmhList)
        toGiveModel.append(classsList)
        #append the initial break value
        toGiveModel.append([0])
        # print(toGiveModel)
        return toGiveModel

    def process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        # Convert Carla image to BGR format
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3] # this is to ignore the 4th Alpha channel - up to 3
        self.front_camera = i

    def process_img_rgb(self, image):
        image.convert(carla.ColorConverter.Raw)
        # Convert Carla image to BGR format
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height_rgb, self.im_width_rgb, 4))[:, :, :3] # this is to ignore the 4th Alpha channel - up to 3
        self.front_camera_rgb = i

        
       
        #save YOLO predictions
        # results = model(i, save=False, conf=0.2, verbose=False)
        # for result in results:
        #     cv2.imwrite(f'E:\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\examples\\B4Pred\\file{self.fileCount}.jpg', i)
        #     result.save(f"E:\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\examples\\AfterPred\\file{self.fileCount}.jpg")
        #     self.fileCount+=1
        #     print(result.boxes.xyxy)
        #     print(result.boxes.cls)
        
    def collision_data(self, event):
        self.collision_hist.append(event)
    # def lane_data(self, event):
    #     self.lane_invade_hist.append(event)
        
        
    
