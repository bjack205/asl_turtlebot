#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Twist, PoseArray, Pose2D
from asl_turtlebot.msg import DetectedObject, Task, RescueInfo, ExploreInfo
import tf
import math
from enum import Enum
import numpy as np

# threshold at which we consider the robot at a location
POS_EPS = .1
THETA_EPS = .15

# time to stop at a stop sign
STOP_TIME = 3

# minimum distance from a stop sign to obey it
STOP_MIN_DIST = 20  # cm

# time taken to cross an intersection
CROSSING_TIME = 3

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

class Supervisor:
    """ the state machine of the turtlebot """

    def __init__(self):
        rospy.init_node('turtlebot_supervisor', anonymous=True)

        # current pose
        self.x = 0
        self.y = 0
        self.theta = 0

        # current mode
        self.mode = Mode.EXPLORE
        self.last_mode_printed = None

        # mission stuff
        self.station_location = (0, 0)
        self.cur_goal = (0, 0)
        self.all_animals_done = False
        self.rescuers_on = False

        # new vars
        self.timer = rospy.Time()
        self.laser_ranges = 0
        self.laser_inc = 0
        self.prev_mode = self.mode
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
        rospy.Subscriber('/explore_info', ExploreInfo, self.explore_callback)
        rospy.Subscriber('/rescue_info', RescueInfo, self.rescue_callback)
        rospy.Subscriber('/rescue_on', String, self.rescue_on_callback)

        # Set up publishers
        self.task_pub = rospy.Publisher('/task', Task, queue_size=10)
        self.pose_goal_publisher = rospy.Publisher('/cmd_pose', Pose2D, queue_size=10)
        self.rescue_pub = rospy.Publisher('/read_to_rescue', String, queue_size=10)

        # TF listener
        self.tf_listener = tf.TransformListener()
        self.tf_broadcast = tf.TransformBroadcaster()

    def animal_detected_callback(self, msg):
        dist = msg.distance
        conf = msg.confidence
        theta = (msg.thetaleft + msg.thetaright) / 2
        theta = theta % 360

    def explore_callback(self):
        a = None

    def rescue_callback(self):
        a = None

    def rescue_on_callback(self):
        a = None
        
    def scan_callback(self, msg):
        self.laser_ranges = msg.ranges
        self.laser_inc = msg.angle_increment

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

    def print_pose(self):
        rospy.loginfo("Current Pose: x=%f.2, y=%f.2, th=%f.2"
                      % (self.x, self.y,  math.degrees(self.theta)))

    def print_goal(self):
        rospy.loginfo("Commanded Pose: x=%f.2, y=%f.2, th=%f.2"
                      % (x_g, y_g,  math.degrees(theta_g)))

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
        self.print_pose()
        # print(pose_world)
        
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
        if obj_class == "stop_sign" and dist < STOP_MIN_DIST and dist > 1 \
           and conf > 0.1 \
           and not self.mode == Mode.CROSS \
           and not self.mode == Mode.STOP:
            self.mode = Mode.STOP
            if not self.mode == Mode.STOP:
                self.prev_mode = self.mode
            else:
                self.prev_mode = Mode.EXPLORE
            self.timer = rospy.get_time()

        ### END OF YOUR CODE ###

    def close_to(self, x, y, theta):
        """ checks if the robot is at a pose within some threshold """

        return (abs(x-self.x)<POS_EPS and abs(y-self.y)<POS_EPS and abs(theta-self.theta)<THETA_EPS)
    def close_to_goal(self):
        """ checks if the robot is at a pose within some threshold """
        return abs(self.cur_goal[0]-self.x)<POS_EPS and  abs(self.cur_goal[1]-self.y)<POS_EPS

    def go_to_goal(self):
        pose_g_msg = Pose2D()
        pose_g_msg.x = self.cur_goal[0]
        pose_g_msg.y = self.cur_goal[1]

    def go_to_pose(self):
        """ sends the current desired pose to the pose controller """

        pose_g_msg = Pose2D()
        pose_g_msg.x = self.cur_goal[0]
        pose_g_msg.y = self.cur_goal[1]
        pose_g_msg.theta = self.cur_goal[2]

        self.pose_goal_publisher.publish(pose_g_msg)

    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """

        self.update_pose()
        self.publish_detections()
        # self.print_pose()
        # self.print_goal()

        # logs the current mode
        if not(self.last_mode_printed == self.mode):
            rospy.loginfo("Current Mode: %s", self.mode)
            self.last_mode_printed = self.mode

        # checks wich mode it is in and acts accordingly
        if self.mode == Mode.TASK_COMPLETE:
            # not doing anything
            rospy.loginfo("Task Complete!!!")
            pass

        elif self.mode == Mode.EXPLORE:
            # moving towards a desired pose
            self.mode == Mode.MANUAL  # For now

        elif self.mode == Mode.GO_TO_STATION:
            self.cur_goal = self.station_location
            if self.close_to_goal():
                self.mode = Mode.WAIT_FOR_RESCUERS
            else:
                self.go_to_goal()

        elif self.mode == Mode.WAIT_FOR_RESCUERS:
            self.rescue_pub.publish("Waiting for rescuers")
            if self.rescuers_on:
                self.mode = Mode.GO_TO_ANIMAL
            else:
                pass

        elif self.mode == Mode.GO_TO_ANIMAL:
            self.cur_goal = (0, 0)  # TODO: Set this to the goal from rescuer node
            if self.close_to_goal():
                self.mode = Mode.ANIMAL_DROPOFF
                self.timer = rospy.get_time()
            elif self.all_animals_done:
                self.mode = Mode.TASK_COMPLETE
            else:
                self.go_to_goal()

        elif self.mode == Mode.ANIMAL_DROPOFF:
            t_elapse = rospy.get_time() - self.timer
            if t_elapse > ANIMAL_DROPOFF_WAIT:
                self.mode = Mode.GO_TO_ANIMAL

        elif self.mode == Mode.STOP:
            # at a stop sign
            t_elapse = rospy.get_time() - self.timer
            if t_elapse > STOP_TIME:
                self.mode = Mode.CROSS
                self.timer = rospy.get_time()

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
        rate = rospy.Rate(5) # 5 Hz
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()


if __name__ == '__main__':
    sup = Supervisor()
    sup.run()
