#!/usr/bin/env python3

import rospy
import shapely
import math
import numpy as np
import threading
from ros_numpy import msgify
from autoware_mini.msg import Path, DetectedObjectArray
from sensor_msgs.msg import PointCloud2
from shapely import LineString, Point

DTYPE = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('vx', np.float32),
    ('vy', np.float32),
    ('vz', np.float32),
    ('distance_to_stop', np.float32),
    ('deceleration_limit', np.float32),
    ('category', np.int32)
])

class CollisionPointsManager:

    def __init__(self):

        # parameters
        self.safety_box_width = rospy.get_param("safety_box_width")
        self.stopped_speed_limit = rospy.get_param("stopped_speed_limit")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")

        # variables
        self.detected_objects = None
        self.goal_point = None

        # Lock for thread safety
        self.lock = threading.Lock()

        # publishers
        self.local_path_collision_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('global_path', Path, self.global_path_callback, queue_size=1, tcp_nodelay=True)

    def global_path_callback(self, msg):

        # get the last point in global path as goal point
        if len(msg.waypoints) > 0:
            self.goal_point = msg.waypoints[-1].position
        else:
            self.goal_point = None

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def path_callback(self, msg):
        collision_points = np.array([], dtype=DTYPE)
        detected_objects = self.detected_objects
        goal_point = self.goal_point

        if len(msg.waypoints) == 0 or detected_objects == None:
            with self.lock:
                collision_points = np.array([], dtype=DTYPE)
                rospy.logwarn_throttle(3, "%s - detected objects not received!", rospy.get_name())
           
        else:
            local_path_linestring = LineString([(waypoint.position.x, waypoint.position.y)for waypoint in msg.waypoints])
            goal_point_point = Point(goal_point.x, goal_point.y)
            local_path_buffer = local_path_linestring.buffer(self.safety_box_width/2, cap_style='flat')
            shapely.prepare(local_path_buffer) #to optimize - faster
        
            with self.lock:
                if goal_point is not None and local_path_buffer.intersects(goal_point_point):
                        goal_data = np.array([(
                            goal_point.x,
                            goal_point.y,
                            goal_point.z,
                            0.0, # vx
                            0.0, # vy
                            0.0, # vz
                            self.braking_safety_distance_goal,
                            np.inf,
                            1 # Category 1
                            )], dtype=DTYPE)
                        collision_points = np.append(collision_points, goal_data)
                
                for obj in detected_objects:
                    object_hull = shapely.polygons(np.array(obj.convex_hull).reshape(-1, 3))

                    if local_path_buffer.intersects(object_hull):
                        intersection = local_path_buffer.intersection(object_hull)
                        intersection_points = shapely.get_coordinates(intersection)
                        object_speed = math.sqrt(obj.velocity.x**2 + obj.velocity.y**2 + obj.velocity.z**2) #obj.velocity - geometry_msgs/Vector3 

                        for x, y in intersection_points:
                            for x, y in intersection_points:
                                collision_points = np.append(collision_points, 
                                                             np.array([(x, y, 
                                                                        obj.centroid.z, 
                                                                        obj.velocity.x, 
                                                                        obj.velocity.y, 
                                                                        obj.velocity.z,
                                                                        self.braking_safety_distance_obstacle, 
                                                                        np.inf, 
                                                                        3 if object_speed < self.stopped_speed_limit else 4)], dtype=DTYPE))

        collision_points_msg = msgify(PointCloud2, collision_points)
        collision_points_msg.header = msg.header
        self.local_path_collision_pub.publish(collision_points_msg)      

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('collision_points_manager')
    node = CollisionPointsManager()
    node.run()