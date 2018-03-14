#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray, String

from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from asl_turtlebot.msg import DetectedObject
import tf
import math
from enum import Enum
import numpy as np
import time

# threshold at which we consider the robot at a location
POS_EPS = .1
THETA_EPS = .3

# time to stop at a stop sign
STOP_TIME = 3

# minimum distance from a stop sign to obey it
STOP_MIN_DIST = 20  # cm

# time taken to cross an intersection
CROSSING_TIME = 10

ANIMAL_DROPOFF_WAIT = 1


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

        # current mode
        #self.mode = Mode.EXPLORE
        # for simulator
        self.mode = Mode.EXPLORE
        self.last_mode_printed = None

        # mission stuff (0.851, 0.308, 3.133), (0.318, 1.483, 1.681), (1.150, 2.791, -0.001), 
        self.station_location = (3.372, 1.698, -1.538)
        self.explore_location = [(3.373, 1.508, -1.619), (3.414, 2.645, -1.585), (3.380, 1.013, -1.536), (2.328, 0.288, -3.093)]
        self.explore_index = 0
        self.animal_location = [(2.346, 1.519, 1.517), (2.454, 2.747, 3.119), (3.371, 2.731, -0.001), (3.372, 1.698, -1.538)]
        self.animal_index = 0

        self.rescuers_on = False
        self.done_crossing = True
        self.cross_time = 0

        # new vars
        self.timer = rospy.Time()
        self.laser_ranges = 0
        self.laser_inc = 0
        self.next_mode = self.mode
        # rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # detections
        self.detections = {}  # dictionary of name ("cat", "dog", "stopsign")
                              # returns a list of current detections
        
        
        # stop signs
        self.signs_detected = 0
        self.sign_locations = []
        self.detection_threshold_dist = 15  # cm
        
        # Set up subscribers
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
        rospy.Subscriber('/detector/cat', DetectedObject, self.animal_detected_callback)
        
        # For Simulation
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)
        
        # For Simulation
        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # TF listener
        self.trans_listener = tf.TransformListener() # tf transform listener
        self.tf_broadcast = tf.TransformBroadcaster()

    def animal_detected_callback(self, msg):
        dist = msg.distance
        conf = msg.confidence
        theta = (msg.thetaleft + msg.thetaright) / 2
        theta = theta % 360
        
    def scan_callback(self, msg):
        self.laser_ranges = msg.ranges
        self.laser_inc = msg.angle_increment

    def matches_detection(self, loc_list, location):
        distance = np.linalg.norm(np.array(loc_list) - location, axis=-1)
        min_dist = np.min(distance)
        if min_dist < self.detection_threshold_dist:
            min_ind = np.argmin(distance)
        else:
            min_ind = -1
        return min_ind

    def detection_in_world(self, rho, theta):
        x_robot = rho*np.cos(theta)
        y_robot = rho*np.sin(theta)
        
        self.tf_broadcast.sendTransform((x_robot/100., y_robot/100., 0),
                                        tf.transformations.quaternion_from_euler(0, 0, 0),
                                        rospy.Time.now(),
                                        "stop_sign",
                                        "/base_footprint")
        
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
        
        # Estimate position in world
        pose_world = self.detection_in_world(dist, theta)
        
        # Match with previous detections
        if name in self.detections:  # check if in dictionary
            min_ind = self.matches_detection(self.detections[name], pose_world)
        else:
            print("Create detections list " + name)
            self.detections[name] = []  # add to list
            min_ind = -1
        if min_ind >= 0:  # detected previously 
            previous_location = self.detections[name][min_ind]
            updated_location = [(pose_world[0] + previous_location[0]) / 2.0,
                                (pose_world[1] + previous_location[1]) / 2.0,
                                average_angles(pose_world[2], previous_location[2])]
            self.detections[name][min_ind] = updated_location
        else:  # add new detection
            self.detections[name].append(pose_world)

    def publish_detections(self):
        for name, detections in self.detections.iteritems():
            for i, detection in enumerate(detections):
                self.tf_broadcast.sendTransform((detection[0]/100, detection[1]/100, 0),
                                                tf.transformations.quaternion_from_euler(0, 0, detection[2]),
                                                rospy.Time.now(),
                                                name + "_" + str(i),
                                                "/map")
        

    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all
        so you shouldn't necessarily stop then """
        self.object_detected(msg)
        
        ### YOUR CODE HERE ###
        obj_class = msg.name
        conf = msg.confidence
        dist = msg.distance
        # print(dist, conf)
        
        if (time.time()-self.cross_time > CROSSING_TIME):
            self.done_crossing = True
        
        if obj_class == "stop_sign" and dist < STOP_MIN_DIST and dist > 1 and self.mode == Mode.NAV and self.done_crossing == True:
            self.done_crossing = False
            self.mode = Mode.STOP

    def rviz_goal_callback(self, msg):
        """ callback for a pose goal sent through rviz """

        self.x_g = msg.pose.position.x
        self.y_g = msg.pose.position.y
        rotation = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        euler = tf.transformations.euler_from_quaternion(rotation)
        self.theta_g = euler[2]

        #self.mode = Mode.NAV
        self.mode = Mode.EXPLORE
    
    def stop_sign_detected_callback_theirs(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        dist = msg.distance

        # if close enough and in nav mode, stop
        if dist > 0 and dist < STOP_MIN_DIST and self.mode == Mode.NAV and self.crossing == 0:
            self.mode = Mode.STOP
        

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

    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """
        self.publish_detections()
        
              
        try:
            (translation, rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            self.x_prev = self.x
            self.y_prev = self.y
            self.theta_prev = self.theta
            
            self.x = translation[0]
            self.y = translation[1]
            euler = tf.transformations.euler_from_quaternion(rotation)
            self.theta = euler[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass
            
        rospy.loginfo("Current Pose: (%f, %f, %f)", self.x, self.y, self.theta)
        rospy.loginfo("Current Goal: (%f, %f, %f)", self.x_g, self.y_g, self.theta_g)           
        

        # logs the current mode
        if not(self.last_mode_printed == self.mode):
            rospy.loginfo("Current Mode: %s", self.mode)
            self.last_mode_printed = self.mode
        
        # checks wich mode it is in and acts accordingly
        if self.mode == Mode.TASK_COMPLETE:
            # not doing anything
            rospy.loginfo("Task Complete!!!")
            pass
            
        # sim modes  
        elif self.mode == Mode.IDLE:
            # send zero velocity
            self.stay_idle()
            self.mode = self.next_mode
            
        elif self.mode == Mode.NAV:
            if self.close_to(self.x_g,self.y_g,self.theta_g):
                self.mode = Mode.IDLE
            else:
                self.nav_to_pose()

        elif self.mode == Mode.EXPLORE:
            # moving towards a desired pose
            self.x_g, self.y_g, self.theta_g = self.explore_location[self.explore_index]
            rospy.loginfo("Exploring location: %d", self.explore_index)
            self.explore_index += 1
            self.next_mode = self.mode
            if self.explore_index == len(self.explore_location):
                self.next_mode = Mode.GO_TO_STATION
            self.mode = Mode.NAV
            

        elif self.mode == Mode.GO_TO_STATION:
            self.x_g, self.y_g, self.theta_g = self.station_location
            self.next_mode = Mode.WAIT_FOR_RESCUERS
            self.mode = Mode.NAV

        elif self.mode == Mode.WAIT_FOR_RESCUERS:
            #self.rescue_pub.publish("Waiting for rescuers")
            rospy.loginfo("Waiting for rescuers")
            time.sleep(10)
            rospy.loginfo("Done Waiting for rescuers")
            self.mode = Mode.GO_TO_ANIMAL
            

        elif self.mode == Mode.GO_TO_ANIMAL:
            self.x_g, self.y_g, self.theta_g = self.animal_location[self.animal_index]
            rospy.loginfo("Rescuing animal: %d", self.animal_index)
            self.animal_index += 1
            self.next_mode = self.mode
            
            if self.animal_index == len(self.animal_location):
                self.next_mode = Mode.TASK_COMPLETE
                
            self.mode = Mode.NAV
                
        elif self.mode == Mode.STOP:
            # at a stop sign
            self.stay_idle()
            time.sleep(STOP_TIME)
            self.cross_time = time.time()
            self.mode = Mode.NAV
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
