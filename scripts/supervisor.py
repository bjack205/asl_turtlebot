#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Twist, PoseArray, Pose2D
from asl_turtlebot.msg import DetectedObject, Task, RescueInfo, ExploreInfo
import tf
import math
from enum import Enum

# threshold at which we consider the robot at a location
POS_EPS = .1
THETA_EPS = .15

# time to stop at a stop sign
STOP_TIME = 3

# minimum distance from a stop sign to obey it
STOP_MIN_DIST = 90

# time taken to cross an intersection
CROSSING_TIME = 3

ANIMAL_DROPOFF_WAIT = 1


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
        
        # Set up subscribers
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
       
        rospy.Subscriber('/explore_info', ExploreInfo, self.explore_callback)
        rospy.Subscriber('/rescue_info', RescueInfo, self.rescue_callback)
        rospy.Subscriber('/rescue_on', String, self.rescue_on_callback)

        # Set up publishers
        self.task_pub = rospy.Publisher('/task', Task, queue_size=10)
        self.pose_goal_publisher = rospy.Publisher('/cmd_pose', Pose2D, queue_size=10)
        self.rescue_pub = rospy.Publisher('/read_to_rescue', String, queue_size=10)

        # TF listener
        self.tf_listener = tf.TransformListener()

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

    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all
        so you shouldn't necessarily stop then """

        ### YOUR CODE HERE ###
        obj_class = msg.name
        conf = msg.confidence
        dist = msg.distance
        # print(dist, conf)
        if obj_class == "stop_sign" and dist > STOP_MIN_DIST \
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
