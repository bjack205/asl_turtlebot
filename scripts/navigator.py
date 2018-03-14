#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from std_msgs.msg import Float32MultiArray, String
import tf
import numpy as np
from numpy import linalg
from utils import wrapToPi
from astar import AStar
from grids import StochOccupancyGrid2D
import scipy.interpolate
import matplotlib.pyplot as plt
import warnings

#from smoothing import LineofSight


# threshold at which navigator switches
# from trajectory to pose control
END_POS_THRESH = .2

# threshold to be far enough into the plan
# to recompute it
START_POS_THRESH = .2

# thereshold in theta to start moving forward when path following
THETA_START_THRESH = 0.09
# P gain on orientation before start
THETA_START_P = 1

# threshold to rotate by when stuck
THETA_ROTATE_THRESH = 0.174

# maximum velocity
V_MAX = .2

# maximim angular velocity
W_MAX = .4

# desired crusing velocity
V_DES = 0.12

# gains of the path follower
KPX = .5
KPY = .5
KDX = 1.5
KDY = 1.5

# smoothing condition (see splrep documentation)
SMOOTH = .01

class Navigator:

    def __init__(self):
        rospy.init_node('turtlebot_navigator', anonymous=True)

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = 0.0
        self.y_g = 0.0
        self.theta_g = 0.0

        self.current_plan = []

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0,0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution =  0.1
        self.plan_horizon = 15

        # variables for the controller
        self.V_prev = 0
        self.V_prev_t = rospy.get_rostime()

        self.nav_path_pub = rospy.Publisher('/cmd_path', Path, queue_size=10)
        self.nav_pose_pub = rospy.Publisher('/cmd_pose', Pose2D, queue_size=10)
        self.nav_pathsp_pub = rospy.Publisher('/cmd_path_sp', PoseStamped, queue_size=10)
        self.nav_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.trans_listener = tf.TransformListener()

        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/map_metadata', MapMetaData, self.map_md_callback)
        rospy.Subscriber('/cmd_nav', Pose2D, self.cmd_nav_callback)

    def cmd_nav_callback(self, data):   
        self.x_g = data.x
        self.y_g = data.y
        self.theta_g = data.theta
        self.run_navigator()

    def map_md_callback(self, msg):
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x,msg.origin.position.y)

    def map_callback(self,msg):
        self.map_probs = msg.data
        #print self.map_probs
        if self.map_width>0 and self.map_height>0 and len(self.map_probs)>0:
            self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  self.map_width,
                                                  self.map_height,
                                                  self.map_origin[0],
                                                  self.map_origin[1],
                                                  8,
                                                  self.map_probs)
                      
            self.occupancy_updated = True

    def close_to_end_location(self):
        return (abs(self.x-self.x_g)<END_POS_THRESH and abs(self.y-self.y_g)<END_POS_THRESH)

    def snap_to_grid(self, x):
        return (self.plan_resolution*round(x[0]/self.plan_resolution), self.plan_resolution*round(x[1]/self.plan_resolution))

    def close_to_start_location(self):
        if len(self.current_plan)>0:
            snapped_current = self.snap_to_grid([self.x, self.y])
            snapped_start = self.snap_to_grid(self.current_plan_start_loc)
            return (abs(snapped_current[0]-snapped_start[0])<START_POS_THRESH and abs(snapped_current[1]-snapped_start[1])<START_POS_THRESH)
        return False

    def run_navigator(self):
        """ computes a path from current state to goal state using A* and sends it to the path controller """

        # makes sure we have a location
        try:
            (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            self.x = translation[0]
            self.y = translation[1]
            euler = tf.transformations.euler_from_quaternion(rotation)
            self.theta = euler[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            self.current_plan = []
            return

        # makes sure we have a map
        if not self.occupancy:
            self.current_plan = []
            return

        # if close to the goal, use the pose_controller instead
        if self.close_to_end_location():
            rospy.loginfo("Navigator: Close to nav goal using pose controller")
            pose_g_msg = Pose2D()
            pose_g_msg.x = self.x_g
            pose_g_msg.y = self.y_g
            pose_g_msg.theta = self.theta_g
            self.nav_pose_pub.publish(pose_g_msg)
            self.current_plan = []
            self.V_prev = 0
            return

        # if there is no plan, we are far from the start of the plan,
        # or the occupancy grid has been updated, update the current plan
        if len(self.current_plan)==0 or not(self.close_to_start_location()) or self.occupancy_updated:

            # use A* to compute new plan
            
            state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
            state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
            x_init = self.snap_to_grid((self.x, self.y))
            x_goal = self.snap_to_grid((self.x_g, self.y_g))
            Atime_start = rospy.get_rostime()
            problem = AStar(state_min,state_max,x_init,x_goal,self.occupancy,self.plan_resolution)

            rospy.loginfo("Navigator: Computing navigation plan")
            if problem.solve():
                # time
                Atime_end = rospy.get_rostime()
                

                    
                
                if len(problem.path) > 3:
                    """
                    # smooth path
                    checkpoint = problem.path[0]
                    currentPoint = problem.path[1]
                    nextPoint = problem.path[2]
                    smooth_path = list(problem.path)
                    print (smooth_path)
                    print checkpoint, nextPoint
                    ind = 2
                    while ind < len(problem.path) - 1:
                        if problem.LineofSight(checkpoint, nextPoint):
                            temp = currentPoint
                            currentPoint = nextPoint
                            nextPoint = problem.path[ind+1]
                            print temp
                            smooth_path.remove(temp)
                            print smooth_path
                        else:
                            print smooth_path
                            checkpoint = currentPoint
                            currentPoint = nextPoint
                            nextPoint = problem.path[ind+1]
                        ind = ind + 1
                    
                         
                    #problem.path = list(smooth_path)
                                      
                    """
                    
                    # curve
                    new_path = []
                    ind = 0
                    if len(problem.path) > 4:
                        while ind < len(problem.path)-2:
                            s1 = np.array(list(problem.path[ind]))
                            s2 = np.array(list(problem.path[ind+1]))
                            s3 = np.array(list(problem.path[ind+2]))
                            #scur = 
                            
                            l1 = s2 - s1
                            l2 = s3 - s2
                            
                            if np.dot(l1, l2) == 0:
                                #curve
                                
                                x = [s1[0], s1[0] + l1[0]/4 + l2[0]/4, s1[0] + l1[0]*2/4 + l2[0]*2/4, s1[0] + l1[0]*3/4 + l2[0]*3/4, s3[0]]
                                y = [s1[1], s1[1] + 0.6*(l1[1]/4 + l2[1]/4), s1[1] + 0.8*(l1[1]*2/4 + l2[1]*2/4), s1[1] + 1.2*(l1[1]*3/4 + l2[1]*3/4), s3[1]]
                                """
                                #x = np.linspace(s1[0], s3[0], 10)
                                #y = np.linspace(s1[1], s3[1], 10)
                                if x[-1] < x[0]:
                                    x = x[::-1]
                                    y = y[::-1]
                                    tck = scipy.interpolate.splrep(x, y, k=3, s=SMOOTH)
                                    xcurve = np.linspace(s1[0], s3[0], 4)
                                    xcurve = xcurve[1:5]
                                    ycurve = scipy.interpolate.splev(xcurve[::-1], tck, der=0)
                                    ycurve = ycurve[::-1]         
                                else:
                                    tck = scipy.interpolate.splrep(x, y, k=3, s=SMOOTH)
                                    xcurve = np.linspace(s1[0], s3[0], 4)
                                    ycurve = scipy.interpolate.splev(xcurve[1:5], tck, der=0)
                                                      
                                
                                #x = np.linspace(0,10,4)
                                #y = np.sin(x)
                                
                                #print x
                                #print y
                                

                                new_points = zip(xcurve.tolist(), ycurve.tolist())
                                """
                                if x[-1] < x[0]:
                                    x = x[::-1]
                                    y = y[::-1]
                                    z = np.polyfit(x, y, 2)      
                                else:
                                    z = np.polyfit(x, y, 2)
                                   
                                f = np.poly1d(z)
                                xcurve = np.linspace(s1[0], s3[0], 4)
                                xcurve = xcurve[1:5]
                                ycurve = [f(i) for i in xcurve]
                                new_points = zip(xcurve, ycurve) 
                                #print new_points
                                
                                new_path.append(problem.path[ind])
                                new_path.extend(new_points)
                                new_path.append(problem.path[ind+2])
                                ind = ind + 2
                            else:
                                new_path.extend([problem.path[ind], problem.path[ind+1], problem.path[ind+2]])
                                ind = ind + 2
                                
                        problem.path = new_path
                    
                    #print new_path
                    #print problem.path
                    
                    
                    
                    # cubic spline interpolation requires 4 points
                    splinetime_start = rospy.get_rostime()
                    self.current_plan = problem.path
                    self.current_plan_start_time = rospy.get_rostime()
                    self.current_plan_start_loc = [self.x, self.y]
                    self.occupancy_updated = False

                    # publish plan for visualization
                    path_msg = Path()
                    path_msg.header.frame_id = 'map'
                    for state in self.current_plan:
                        pose_st = PoseStamped()
                        pose_st.pose.position.x = state[0]
                        pose_st.pose.position.y = state[1]
                        pose_st.pose.orientation.w = 1
                        pose_st.header.frame_id = 'map'
                        path_msg.poses.append(pose_st)
                    self.nav_path_pub.publish(path_msg)

                    path_t = [0]
                    path_x = [self.current_plan[0][0]]
                    path_y = [self.current_plan[0][1]]
                    for i in range(len(self.current_plan)-1):
                        dx = self.current_plan[i+1][0]-self.current_plan[i][0]
                        dy = self.current_plan[i+1][1]-self.current_plan[i][1]
                        path_t.append(path_t[i] + np.sqrt(dx**2 + dy**2) / V_DES)
                        path_x.append(self.current_plan[i+1][0])
                        path_y.append(self.current_plan[i+1][1])

                    # interpolate the path with cubic spline
                    #print path_t
                    #print path_y
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error')
                        try:
                            self.path_x_spline = scipy.interpolate.splrep(path_t, path_x, k=3, s=SMOOTH)
                            self.path_y_spline = scipy.interpolate.splrep(path_t, path_y, k=3, s=SMOOTH)
                        except Warning as e:
                            print('error found:', e)
                            print('disabling smooting interpolation')
                            self.path_x_spline = scipy.interpolate.splrep(path_t, path_x, k=3, s=0)
                            self.path_y_spline = scipy.interpolate.splrep(path_t, path_y, k=3, s=0)                            

                    self.path_tf = path_t[-1]
                    
                    splinetime_end = rospy.get_rostime()
                    rospy.loginfo("cubic time: %f", (splinetime_end - splinetime_start).to_sec())

                    # to inspect the interpolation and smoothing
                    # t_test = np.linspace(path_t[0],path_t[-1],1000)
                    # plt.plot(path_t,path_x,'ro')
                    # plt.plot(t_test,scipy.interpolate.splev(t_test,self.path_x_spline,der=0))
                    # plt.plot(path_t,path_y,'bo')
                    # plt.plot(t_test,scipy.interpolate.splev(t_test,self.path_y_spline,der=0))
                    # plt.show()
                    #
                else:
                    rospy.logwarn("Navigator: Path too short, not updating")
            else:
                rospy.logwarn("Navigator: Could not find path")
                # rotate around in place
                cmd_msg = Twist()
                cmd_msg.linear.x = 0
                cmd_msg.angular.z = self.theta + THETA_ROTATE_THRESH
                self.nav_vel_pub.publish(cmd_msg)
                self.current_plan = []

        # if we have a path, execute it (we need at least 3 points for this controller)
        if len(self.current_plan) > 3:

            # if currently not moving, first line up with the plan
            
            if self.V_prev == 0:
                print ("aligninng heading")
                theta_init = np.arctan2(self.current_plan[1][1]-self.current_plan[0][1],self.current_plan[1][0]-self.current_plan[0][0])
                theta_err = theta_init-self.theta
                if abs(theta_err)>THETA_START_THRESH:
                    cmd_msg = Twist()
                    cmd_msg.linear.x = 0
                    cmd_msg.angular.z = THETA_START_P * theta_err
                    self.nav_vel_pub.publish(cmd_msg)
                    return
            

            # compute the "current" time along the path execution
            t = (rospy.get_rostime()-self.current_plan_start_time).to_sec()
            t = max(0.0, t)
            t = min(t, self.path_tf)

            x_d = scipy.interpolate.splev(t, self.path_x_spline, der=0)
            y_d = scipy.interpolate.splev(t, self.path_y_spline, der=0)
            xd_d = scipy.interpolate.splev(t, self.path_x_spline, der=1)
            yd_d = scipy.interpolate.splev(t, self.path_y_spline, der=1)
            xdd_d = scipy.interpolate.splev(t, self.path_x_spline, der=2)
            ydd_d = scipy.interpolate.splev(t, self.path_y_spline, der=2)

            # publish current desired x and y for visualization only
            pathsp_msg = PoseStamped()
            pathsp_msg.header.frame_id = 'map'
            pathsp_msg.pose.position.x = x_d
            pathsp_msg.pose.position.y = y_d
            theta_d = np.arctan2(yd_d,xd_d)
            quat_d = tf.transformations.quaternion_from_euler(0, 0, theta_d)
            pathsp_msg.pose.orientation.x = quat_d[0]
            pathsp_msg.pose.orientation.y = quat_d[1]
            pathsp_msg.pose.orientation.z = quat_d[2]
            pathsp_msg.pose.orientation.w = quat_d[3]
            self.nav_pathsp_pub.publish(pathsp_msg)

            if self.V_prev <= 0.0001:
                self.V_prev = linalg.norm([xd_d, yd_d])

            dt = (rospy.get_rostime()-self.V_prev_t).to_sec()

            xd = self.V_prev*np.cos(self.theta)
            yd = self.V_prev*np.sin(self.theta)

            u = np.array([xdd_d + KPX*(x_d-self.x) + KDX*(xd_d-xd),
                          ydd_d + KPY*(y_d-self.y) + KDY*(yd_d-yd)])
            J = np.array([[np.cos(self.theta), -self.V_prev*np.sin(self.theta)],
                          [np.sin(self.theta), self.V_prev*np.cos(self.theta)]])
            a, om = linalg.solve(J, u)
            V = self.V_prev + a*dt

            # apply saturation limits
            cmd_x_dot = np.sign(V)*min(V_MAX, np.abs(V))
            cmd_theta_dot = np.sign(om)*min(W_MAX, np.abs(om))
        elif len(self.current_plan) > 0:
            # using the pose controller for paths too short
            # just send the next point
            pose_g_msg = Pose2D()
            pose_g_msg.x = self.current_plan[0][0]
            pose_g_msg.y = self.current_plan[0][1]
            if len(self.current_plan)>1:
                pose_g_msg.theta = np.arctan2(self.current_plan[1][1]-self.current_plan[0][1],self.current_plan[1][0]-self.current_plan[0][0])
            else:
                pose_g_msg.theta = self.theta_g
            self.nav_pose_pub.publish(pose_g_msg)
            return
        else:
            # just stop
            cmd_x_dot = 0
            cmd_theta_dot = 0

        # saving the last velocity for the controller
        self.V_prev = cmd_x_dot
        self.V_prev_t = rospy.get_rostime()

        cmd_msg = Twist()
        cmd_msg.linear.x = cmd_x_dot
        cmd_msg.angular.z = cmd_theta_dot
        self.nav_vel_pub.publish(cmd_msg)


if __name__ == '__main__':
    nav = Navigator()
    rospy.spin()
