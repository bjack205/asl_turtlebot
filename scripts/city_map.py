#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path

INCHES_IN_METER = 39.37

CITY = [
  # outside border
  ((0, 0), (0, 84)),
  ((0, 84), (1.761965413374476, 95.1246117974981)),
  ((1.761965413374476, 95.1246117974981), (6.875388202501895, 105.16026908252904)),
  ((6.875388202501895, 105.16026908252904), (14.83973091747097, 113.1246117974981)),
  ((14.83973091747097, 113.1246117974981), (24.875388202501895, 118.23803458662553)),
  ((24.875388202501895, 118.23803458662553), (36.0, 120.0)),
  ((36.0, 120.0), (144, 120)),
  ((144, 120), (144, 0)),
  ((144, 0), (0, 0)),
  # convex 1
  ((22, 22), (22, 52)),
  ((22, 52), (81, 52)),
  ((81, 52), (81, 22)),
  ((81, 22), (22, 22)),
  # convex 2
  ((22, 69), (22, 84)),
  ((22, 84), (22.685208771867849, 88.326237921249259)),
  ((22.685208771867849, 88.326237921249259), (24.673762078750737, 92.22899353209462)),
  ((24.673762078750737, 92.22899353209462), (27.77100646790538, 95.326237921249259)),
  ((27.77100646790538, 95.326237921249259), (31.673762078750737, 97.314791228132151)),
  ((31.673762078750737, 97.314791228132151), (36.0, 98.0)),
  ((36, 98), (86, 98)),
  ((86, 98), (86, 90)),
  ((86, 90), (73, 76)),
  ((73, 76), (73, 69)),
  ((73, 69), (22, 69)),
  # convex 3
  ((103, 98), (123, 98)),
  ((123, 22), (103, 22)),
  ((103, 22), (103, 98)),
]

CITY = [((s[0][0]/INCHES_IN_METER, s[0][1]/INCHES_IN_METER),
         (s[1][0]/INCHES_IN_METER, s[1][1]/INCHES_IN_METER)) for s in CITY]

LANE_LINES = [((133, 109), (133, 11)),
              ((133, 11), (11, 11)),
              ((11, 11), (11, 84)),
              ((11, 60.5), (92, 60.5)),
              ((92, 11), (92, 109)),
              ((92, 11), (92, 109)),
              ((36, 109), (133, 109))]

LANE_LINES_DASHED = []
for L in LANE_LINES:
    p0 = np.array(L[0])
    p1 = np.array(L[1])
    N = int(np.linalg.norm(p0 - p1) / 3.0)
    if N % 2 == 1:
        N = N + 1
    a = np.linspace(0, 1, N)
    LANE_LINES_DASHED = LANE_LINES_DASHED + [(p0*a[i] + p1*(1-a[i]), p0*a[i+1] + p1*(1-a[i+1])) for i in range(0,N,2)]


c = np.array([36.0, 84.0])
r = 25.0
N = int((r*np.pi/2) / 3.0)
if N % 2 == 1:
    N = N + 1
a = np.linspace(0, 1, N)*np.pi/2 + np.pi/2
p = [(c + np.array([r*np.cos(t), r*np.sin(t)])) for t in a]
LANE_LINES_DASHED = LANE_LINES_DASHED + [((p[i], p[i+1])) for i in range(0,N,2)]

LANE_LINES_DASHED = [((s[0][0]/INCHES_IN_METER, s[0][1]/INCHES_IN_METER),
                      (s[1][0]/INCHES_IN_METER, s[1][1]/INCHES_IN_METER)) for s in LANE_LINES_DASHED]


class LineSegmentVisualizer:

    def __init__(self):
        rospy.init_node('line_segment_viz')

        self.ground_truth_map_pub = rospy.Publisher("ground_truth_map", Marker, queue_size=10)
        self.ground_truth_map_marker = Marker()
        self.ground_truth_map_marker.header.frame_id = "/map"
        self.ground_truth_map_marker.header.stamp = rospy.Time.now()
        self.ground_truth_map_marker.ns = "ground_truth"
        self.ground_truth_map_marker.type = 5    # line list
        self.ground_truth_map_marker.pose.orientation.w = 1.0
        self.ground_truth_map_marker.scale.x = .025
        self.ground_truth_map_marker.scale.y = .025
        self.ground_truth_map_marker.scale.z = .025
        self.ground_truth_map_marker.color.r = 0.0
        self.ground_truth_map_marker.color.g = 1.0
        self.ground_truth_map_marker.color.b = 0.0
        self.ground_truth_map_marker.color.a = 1.0
        self.ground_truth_map_marker.lifetime = rospy.Duration(1000)
        self.ground_truth_map_marker.points = sum([[Point(p1[0], p1[1], 0),
                                                    Point(p2[0], p2[1], 0)] for p1, p2 in CITY], [])

        self.lane_lines_map_pub = rospy.Publisher("lane_lines_map", Marker, queue_size=10)
        self.lane_lines_map_marker = Marker()
        self.lane_lines_map_marker.header.frame_id = "/map"
        self.lane_lines_map_marker.header.stamp = rospy.Time.now()
        self.lane_lines_map_marker.ns = "lane_lines"
        self.lane_lines_map_marker.type = 5    # line list
        self.lane_lines_map_marker.pose.orientation.w = 1.0
        self.lane_lines_map_marker.scale.x = .025
        self.lane_lines_map_marker.scale.y = .025
        self.lane_lines_map_marker.scale.z = .025
        self.lane_lines_map_marker.color.r = 1.0
        self.lane_lines_map_marker.color.g = 0.0
        self.lane_lines_map_marker.color.b = 0.0
        self.lane_lines_map_marker.color.a = 1.0
        self.lane_lines_map_marker.lifetime = rospy.Duration(1000)
        self.lane_lines_map_marker.points = sum([[Point(p1[0], p1[1], 0),
                                                  Point(p2[0], p2[1], 0)] for p1, p2 in LANE_LINES_DASHED], [])
        
        """
        outer_border = CITY[0:9]                                          
        for j in outer_border:
            x_width.extend([j[0][0], j[1][0]])
            y_height.extend([j[0][1], j[1][1]])
        width = max(x_width)
        height = max(y_height)                                          
        self.sim_map_pub = rospy.Publisher("sim_map", MapMetaData, queue_size=10)
        self.sim_map = MapMetaData()
        self.sim_map.width = width
        self.sim_map.height = height
        self.sim_map.resolution = 1e-3
        self.sim_map.origin.position.x = 0
        self.sim_map.origin.position.y = 0
        
        self.sim_map.probs = 0.1
        """


    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.ground_truth_map_pub.publish(self.ground_truth_map_marker)
            self.lane_lines_map_pub.publish(self.lane_lines_map_marker)
            #self.sim_map_pub.publish(self.sim_map)
            rate.sleep()

if __name__ == '__main__':
    vis = LineSegmentVisualizer()
    vis.run()
