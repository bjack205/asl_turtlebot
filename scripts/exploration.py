#!/usr/bin/env python

import rospy

from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from sensor_msgs.msg import LaserScan
from asl_turtlebot.msg import DetectedObject
import tf
import math
import numpy as np
from enum import Enum

# threshold at which we consider the robot at a location
POS_EPS = .05
THETA_EPS = .1

DIS_TO_WALL = 0.35


# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 1
    NAV = 2
    TURN = 3


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
        self.mode = Mode.NAV
        self.last_mode_printed = None

		#pub/sub
        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
        self.pose_goal_publisher = rospy.Publisher('/cmd_pose', Pose2D, queue_size=10)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.trans_listener = tf.TransformListener()
        rospy.Subscriber('/scan', LaserScan, self.dist_to_wall_callback)

        # exploration variables
        self.tf_update = False
        #self.move_list = ["straight","straight","straight"]
        self.turn_list = ["straight","right","right"]
        self.explore_ind = 0
        self.turn_vector = {}
        self.turn_vector["straight"] = np.array([1.0, 0.0, 0.0])
        #self.turn_vector["right"] = np.array([0, -1.0, 0.0])
        #self.turn_vector["left"] = np.array([0, 1.0, 0.0])
        self.distance = 1.0
        self.new_goal = True

    def dist_to_wall_callback(self, data):
        self.distance = np.mean(np.concatenate((data.ranges[0:4],data.ranges[-5:-1])))
        rospy.loginfo("distance to wall: %s", self.distance)

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

    def close_to(self,x,y,theta):
        """ checks if the robot is at a pose within some threshold """

        return (abs(x-self.x)<POS_EPS and abs(y-self.y)<POS_EPS and abs(theta-self.theta)<THETA_EPS)


    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """

        try:
            (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            self.x = translation[0]
            self.y = translation[1]
            euler = tf.transformations.euler_from_quaternion(rotation)
            self.theta = euler[2]
            self.tf_update = True
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            self.tf_update = False
        rospy.loginfo("x: %.3f", self.x)
        rospy.loginfo("y: %.3f", self.y)
        rospy.loginfo("theta: %.3f", self.theta)

        # logs the current mode
        if not(self.last_mode_printed == self.mode):
            rospy.loginfo("Current Mode: %s", self.mode)
            self.last_mode_printed = self.mode

        # checks wich mode it is in and acts accordingly
        if self.mode == Mode.IDLE:
            # send zero velocity
            self.stay_idle()
            self.mode = Mode.NAV
            #self.explore_ind += 1
            #rospy.loginfo("On step: %i", self.explore_ind)

        elif self.mode == Mode.NAV:
			if self.tf_update == True:
				if self.new_goal and self.explore_ind < len(self.turn_list):
					R = np.array([[np.cos(self.theta), -np.sin(self.theta), 0],[np.sin(self.theta), np.cos(self.theta),0],[0,0,1]])
					pos_goal_baseframe = (self.distance - DIS_TO_WALL)*self.turn_vector["straight"]
					pos_goal_worldframe = R.dot(pos_goal_baseframe)
					self.x_g = self.x + pos_goal_worldframe[0]
					self.y_g = self.y + pos_goal_worldframe[1]
					'''
					if self.turn_list[self.explore_ind] == "straight":
					    angle = 0
					elif self.turn_list[self.explore_ind] == "right":
					    angle = -1.5708 # 90 degrees cw
					elif self.turn_list[self.explore_ind] == "left":
					    angle = 1.5708 # 90 degrees ccw
				    '''
					self.theta_g = self.theta -np.pi/2
					self.new_goal = False
					rospy.loginfo("goal created")
				if self.close_to(self.x_g,self.y_g,self.theta_g): #or self.distance <= DIS_TO_WALL:
				    self.mode = Mode.IDLE
				    self.new_goal = True
				else:
				    self.nav_to_pose()
		
		
        elif self.mode == Mode.TURN:
            if self.tf_update == True:
                if self.new_goal:
                    self.theta_g = self.theta - 1.5708
                    self.new_goal = False
                if self.close_to(self.x_g,self.y_g,self.theta_g):
                    self.mode = Mode.IDLE
                    self.new_goal = True
                else:
                    self.nav_to_pose()
		    

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
