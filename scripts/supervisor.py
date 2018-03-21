#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray, String, Bool

from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from asl_turtlebot.msg import DetectedObject, Task, RescueInfo, ExploreInfo
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
import tf
import math
from enum import Enum
import numpy as np
from explore import *


# threshold at which we consider the robot at a location
POS_EPS = .2  # For animal collection
THETA_EPS = .3

# time to stop at a stop sign
STOP_TIME = 3

# minimum distance from a stop sign to obey it
STOP_MIN_DIST = 20  # cm

# time taken to cross an intersection
CROSSING_TIME = 3
ANIMAL_DROPOFF_WAIT = 1

def angdiff(ang1, ang2):
    temp = np.abs(ang1 - ang2)
    if temp > np.pi:
        temp = 2*np.pi - temp
    temp = temp % (2*np.pi)
    return temp

def average_angles(ang1, ang2):
    ang1 += np.pi
    ang2 += np.pi
    ang1 %= 2*np.pi
    ang2 %= 2*np.pi
    avg = (ang1 + ang2) / 2
    return (avg - np.pi) % (2*np.pi)


# state machine modes, not all implemented
class Mode(Enum):
    EXPLORE = 1
    GO_TO_STATION = 2
    WAIT_FOR_RESCUERS = 3
    GO_TO_ANIMAL = 4
    ANIMAL_DROPOFF = 5
    TASK_COMPLETE = 6
    STOP = 7
    CROSS = 8
    MANUAL = 9
    # SIM
    IDLE = 10
    NAV = 11

class Supervisor:
    """ the state machine of the turtlebot """

    def __init__(self):
        rospy.init_node('turtlebot_supervisor', anonymous=True)

        # current pose
        self.x = 0
        self.y = 0
        self.theta = 0

        # pose goal
        self.x_g = 0
        self.y_g = 0
        self.theta_g = 0
        
        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0,0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False
        
        self.world_width = 3.0
        self.world_height = 3.0

        # current mode
        #self.mode = Mode.EXPLORE
        # for simulator
        self.mode = Mode.EXPLORE
        self.last_mode_printed = None

        # mission stuff
        self.station_location = (0, 0)
        self.cur_goal = (0, 0)
        self.all_animals_done = False
        self.rescuers_on = False
        self.is_crossing = False  # Flag if it passing a stop sign
        self.temp1 = True
        self.cur_animal = 0
        self.dropoff_timer = 0
        
        # new vars
        self.timer = rospy.Time()
        self.laser_ranges = 0
        self.laser_inc = 0
        self.prev_mode = self.mode
        # rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # detections
        self.detections = {}  # dictionary of name ("cat", "dog", "stop_sign")
                              # returns a list of current detections in the world framex
        
        # stop signs
        self.signs_detected = 0
        self.sign_locations = []
        self.detection_threshold_dist = 20.0/100.0  # m
        self.sign_stop_distance = 10.0/100.0  # Distance to esimated stop sign to stop (cm)
        self.sign_stop_angle = np.radians(90)  # Angle difference to stop sign to stop (rad)
        self.stop_sign_start = rospy.get_rostime()
        
        # Set up subscribers
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
        rospy.Subscriber('/detector/cat', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/detector/dog', DetectedObject, self.animal_detected_callback)
        rospy.Subscriber('/explore_info', ExploreInfo, self.explore_callback)
        rospy.Subscriber('/rescue_info', RescueInfo, self.rescue_callback)
        rospy.Subscriber('/ready_to_rescue', Bool, self.rescue_on_callback)
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/map_metadata', MapMetaData, self.map_md_callback)
        
        # For Simulation
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)

        # Set up publishers
        self.task_pub = rospy.Publisher('/task', Task, queue_size=10)
        self.pose_goal_publisher = rospy.Publisher('/cmd_pose', Pose2D, queue_size=10)
        self.rescue_pub = rospy.Publisher('/ready_to_rescue', Bool, queue_size=10)
        
        # For Simulation
        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
        self.cmd_vel_publisher = rospy.Publisher('/nav_vel', Twist, queue_size=10)

        # TF listener
        self.tf_listener = tf.TransformListener()
        self.tf_broadcast = tf.TransformBroadcaster()
        
    def map_md_callback(self, msg):
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x,msg.origin.position.y)

    def map_callback(self,msg):
        self.map_probs = msg.data
        if self.map_width>0 and self.map_height>0 and len(self.map_probs)>0:
            self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  self.map_width,
                                                  self.map_height,
                                                  self.map_origin[0],
                                                  self.map_origin[1],
                                                  8,
                                                  self.map_probs)
            self.occupancy_updated = True

    def animal_detected_callback(self, msg):
        msg.name = 'animal'
        self.object_detected(msg)
        
    def explore_callback(self,msg):
        if msg.info == "yes":
            #self.x_g = 0#msg.goal[0]
            #self.y_g = 0#msg.goal[1]
            #self.theta_g = self.theta
            self.mode = Mode.NAV
            rospy.loginfo('MSG Received: STOP EXPLORING')

    def rescue_callback(self):
        pass

    def rescue_on_callback(self, msg):
        if msg.data and (self.mode == Mode.NAV or self.mode == Mode.IDLE or self.mode == Mode.EXPLORE):
            self.mode = Mode.GO_TO_ANIMAL
        
    def print_pose(self):
        rospy.loginfo("Current Pose: x=%f.2, y=%f.2, th=%f.2"
                      % (self.x, self.y,  math.degrees(self.theta)))

    def print_goal(self):
        rospy.loginfo("Commanded Pose: x=%.2f, y=%.2f, th=%.2f"
                      % (self.x_g, self.y_g,  math.degrees(self.theta_g)))

    def matches_detection(self, loc_list, location):
        distance = []
        ang_dif = []
        for ind in range(len(loc_list)):
            dist = np.linalg.norm(np.array(loc_list[ind][0:2]) - np.array(location[0:2]))
            distance.append(dist)
            ang_dif.append(angdiff(loc_list[ind][2], location[2]))
            
        min_dist = np.min(distance)
        min_ind = np.argmin(distance)
        min_ang = ang_dif[min_ind]
        
        if min_dist <  self.detection_threshold_dist:
            return min_ind
        else:
            min_ind = -1
        return min_ind

    def detection_in_world(self, rho, theta):
        x_robot = rho*np.cos(theta)
        y_robot = rho*np.sin(theta)
        x_robot /= 100
        y_robot /= 100
        '''
        self.tf_broadcast.sendTransform((x_robot/100., y_robot/100., 0),
                                        tf.transformations.quaternion_from_euler(0, 0, 0),
                                        rospy.Time.now(),
                                        "stop_sign",
                                        "/base_footprint")
        '''
        
        x_world = x_robot*np.cos(self.theta) - y_robot*np.sin(self.theta) + self.x
        y_world = y_robot*np.cos(self.theta) + x_robot*np.sin(self.theta) + self.y
        theta_world = self.theta  # Estimate orientation of object as that of the robot
        return x_world, y_world, theta_world

    def object_detected(self, msg):
        # Get info from message
        name = msg.name
        dist = msg.distance
        conf = msg.confidence
        theta = average_angles(msg.thetaleft, msg.thetaright)
        # print(np.degrees(msg.thetaleft), np.degrees(msg.thetaright), np.degrees(theta))
        
        # Estimate position in world
        pose_world = self.detection_in_world(dist, theta)
        # self.print_pose()
        # print(pose_world)
        
        # Matc hwith previous detections
        if name in self.detections:  # check if in dictionary
            min_ind = self.matches_detection(self.detections[name], pose_world)
        else:
            self.detections[name] = []  # add to list
            min_ind = -1
            
        if min_ind >= 0:  # detected previously
            previous_location = self.detections[name][min_ind]
            # TODO: Replace this with EKF?
            updated_location = [(pose_world[0] + previous_location[0]) / 2.0,
                                (pose_world[1] + previous_location[1]) / 2.0,
                                average_angles(pose_world[2], previous_location[2])]
            self.detections[name][min_ind] = updated_location
        else:  # add new detection
            self.detections[name].append(list(pose_world))

    def publish_detections(self):
        for name, detections in self.detections.iteritems():
            for i, detection in enumerate(detections):
                self.tf_broadcast.sendTransform(
                    (detection[0], detection[1], 0),
                    tf.transformations.quaternion_from_euler(0, 0, detection[2]),
                    rospy.Time.now(),
                    name + "_" + str(i),
                    "/map")

    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all
        so you shouldn't necessarily stop then """
        self.object_detected(msg)
        
    def close_to_stopsign(self):
        if 'stop_sign' in self.detections:
            for idx, sign in enumerate(self.detections['stop_sign']):
                dist = np.sqrt((self.x - sign[0])**2 + (self.y - sign[1])**2)
                dtheta = angdiff(self.theta, sign[2])
                if dist < self.sign_stop_distance and \
                   dtheta < self.sign_stop_angle and \
                   not self.is_crossing:
                    return True
        return False

    def rviz_goal_callback(self, msg):
        """ callback for a pose goal sent through rviz """

        self.x_g = msg.pose.position.x
        self.y_g = msg.pose.position.y
        rotation = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        euler = tf.transformations.euler_from_quaternion(rotation)
        self.theta_g = euler[2]

        self.mode = Mode.NAV
    
    def stop_sign_detected_callback_theirs(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        dist = msg.distance

        # if close enough and in nav mode, stop
        if dist > 0 and dist < STOP_MIN_DIST and self.mode == Mode.NAV:
            self.init_stop_sign()
        

    def go_to_pose(self):
        """ sends the current desired pose to the pose controller """
        pose_g_msg = Pose2D()
        pose_g_msg.x = self.x_g
        pose_g_msg.y = self.y_g
        pose_g_msg.theta = self.theta_g

        self.pose_goal_publisher.publish(pose_g_msg)

    def nav_to_pose(self):
        """ sends the current desired pose to the naviagtor """
        nav_g_msg = Pose2D()
        nav_g_msg.x = self.x_g
        nav_g_msg.y = self.y_g
        nav_g_msg.theta = self.theta_g

        self.nav_goal_publisher.publish(nav_g_msg)

    def stay_idle(self):
        """ sends zero velocity to stay put """
        vel_g_msg = Twist()
        self.cmd_vel_publisher.publish(vel_g_msg)

    def close_to(self, x, y, theta):
        """ checks if the robot is at a pose within some threshold """
        return (abs(x-self.x)<POS_EPS and abs(y-self.y)<POS_EPS and abs(theta-self.theta)<THETA_EPS)

    def close_to_goal(self, x_g, y_g):
        """ checks if the robot is at a pose within some threshold """
        return abs(x_g-self.x)<POS_EPS and abs(y_g-self.y)<POS_EPS

    def go_to_goal(self):
        pose_g_msg = Pose2D()
        pose_g_msg.x = self.cur_goal[0]
        pose_g_msg.y = self.cur_goal[1]

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """

        # pose_g_msg = Pose2D()
        # pose_g_msg.x = self.cur_goal[0]
        # pose_g_msg.y = self.cur_goal[1]
        # pose_g_msg.theta = self.cur_goal[2]
        self.stop_sign_start = rospy.get_rostime()
        if not self.mode == Mode.STOP:
            self.prev_mode = self.mode
        else:
            rospy.loginfo("STOP Loop. Resetting to EXPLORE")
            self.prev_mode = Mode.EXPLORE
        self.mode = Mode.STOP

    def has_stopped(self):
        """ checks if stop sign maneuver is over """
        return (self.mode == Mode.STOP and (rospy.get_rostime()-self.stop_sign_start)>rospy.Duration.from_sec(STOP_TIME))

    def init_crossing(self):
        """ initiates an intersection crossing maneuver """
        self.cross_start = rospy.get_rostime()
        self.is_crossing = True
        self.mode = self.prev_mode

    def has_crossed(self):
        """ checks if crossing maneuver is over """
        if self.is_crossing and \
           (rospy.get_rostime() - self.cross_start) > rospy.Duration.from_sec(CROSSING_TIME):
            self.is_crossing = False
            return True
        return False

    def update_pose(self):
        try:
            trans, rot = self.tf_listener.lookupTransform("/map",
                                                          "/base_footprint",
                                                          rospy.Time(0))
            self.x = trans[0]
            self.y = trans[1]
            _, _, self.theta = tf.transformations.euler_from_quaternion(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), it takes appropriate
        actions. This function shouldn't return anything """

        self.update_pose()
        self.publish_detections()
        # rospy.loginfo(self.detections)
        # self.print_pose()
        # self.print_goal()

        # logs the current mode
        if not(self.last_mode_printed == self.mode):
            rospy.loginfo("Current Mode: %s", self.mode)
            self.last_mode_printed = self.mode

        if self.close_to_stopsign():
            self.init_stop_sign()

        self.has_crossed()

        # checks wich mode it is in and acts accordingly
        if self.mode == Mode.TASK_COMPLETE:
            pass

        # sim modes  
        elif self.mode == Mode.IDLE:
            # send zero velocity
            self.stay_idle()
            # rospy.loginfo("YOU MADE IT!; Dog location:")
            # rospy.loginfo(self.detections['dog'][0])
        
        elif self.mode == Mode.NAV:
            
            # if self.temp1:
                #R = np.array([[np.cos(self.theta), -np.sin(self.theta), 0],[np.sin(self.theta), np.cos(self.theta),0],[0,0,1]])
                #pos_goal_worldframe = R.dot(np.array([1.0, 0.0, 0.0]))
                # self.x_g = 0#self.detections['dog'][0][0] 
                # self.y_g = 0#self.detections['dog'][0][1]
                # self.theta_g = self.theta
                # self.temp1 = False
            
            if self.close_to(self.x_g,self.y_g,self.theta_g):
                self.mode = Mode.IDLE
            else:
                self.nav_to_pose()

        elif self.mode == Mode.EXPLORE:
            if self.occupancy_updated:
                grid_x = int((self.x - self.map_origin[0]) / self.map_resolution)
                grid_y = int((self.y - self.map_origin[1]) / self.map_resolution)
                location = (self.x - self.map_origin[0], self.y - self.map_origin[1])
                rmap = np.array(self.map_probs).reshape(self.map_height, self.map_width)
                
                ### reduce region to explore
                x_limit = int(self.world_height/self.map_resolution)
                y_limit = int(self.world_width/self.map_resolution)
                print(x_limit,y_limit)
                print(grid_x,grid_y)
                rmap = rmap[int(0.98*grid_x):grid_x+x_limit,int(0.98*grid_y):grid_y+y_limit]
               
                if self.close_to(self.x_g,self.y_g,self.theta_g):
                    (self.x_g, self.y_g) = find_explore_location(rmap, location, self.map_width*self.map_resolution, self.map_height*self.map_resolution, self.map_resolution)
                    self.theta_g = self.theta
                else:
                    self.nav_to_pose()
	            

        elif self.mode == Mode.GO_TO_STATION:
            self.cur_goal = self.station_location
            if self.close_to_goal():
                self.mode = Mode.WAIT_FOR_RESCUERS
            else:
                self.nav_to_pose()

        elif self.mode == Mode.WAIT_FOR_RESCUERS:
            self.rescue_pub.publish("Waiting for rescuers")
            if self.rescuers_on:
                self.mode = Mode.GO_TO_ANIMAL
            else:
                pass

        elif self.mode == Mode.GO_TO_ANIMAL:
            if 'animal' in self.detections:
                cur_goal = self.detections['animal'][self.cur_animal]
                self.x_g = cur_goal[0]
                self.y_g = cur_goal[1]
                self.theta_g = cur_goal[2]
                if self.close_to_goal(cur_goal[0], cur_goal[1]):
                    self.mode = Mode.ANIMAL_DROPOFF
                    self.dropoff_timer = rospy.get_time()
                    self.cur_animal += 1
                else:
                    self.nav_to_pose()
                if self.cur_animal >= len(self.detections['animal']):
                    self.mode = Mode.TASK_COMPLETE
            else:
                self.mode = Mode.TASK_COMPLETE


        elif self.mode == Mode.ANIMAL_DROPOFF:
            t_elapse = rospy.get_time() - self.dropoff_timer
            if t_elapse > ANIMAL_DROPOFF_WAIT:
                self.mode = Mode.GO_TO_ANIMAL

        elif self.mode == Mode.STOP:
            # at a stop sign
            if self.has_stopped():
                self.init_crossing()

        elif self.mode == Mode.CROSS:
            # crossing an intersection
            self.go_to_goal()
            t_elapse = rospy.get_time() - self.timer
            if t_elapse > CROSSING_TIME:
                self.mode = self.prev_mode
        else:
            raise Exception('This mode is not supported: %s'
                % str(self.mode))

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()


if __name__ == '__main__':
    sup = Supervisor()
    sup.run()
