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
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from autoware_mini.msg import TrafficLightResult, TrafficLightResultArray

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
        self.braking_safety_distance_stopline = rospy.get_param("~braking_safety_distance_stopline")

        # Parameters related to lanelet2 map loading
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        lanelet2_map_path = rospy.get_param("~lanelet2_map_path")

        # variables
        self.detected_objects = None
        self.goal_point = None
        self.stopline_statuses = None

        # Lock for thread safety
        self.lock = threading.Lock()

        # Load the map using Lanelet2
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
        lanelet2_map = load(lanelet2_map_path, projector)

        # Extract all stop lines and signals from the lanelet2 map
        self.all_stoplines = get_stoplines(lanelet2_map)

        # publishers
        self.local_path_collision_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('global_path', Path, self.global_path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/traffic_light_status', TrafficLightResultArray, self.traffic_light_status_callback, queue_size=1, tcp_nodelay=True)

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
        stopline_statuses = self.stopline_statuses
        stopline_point = []

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
                
                if len(stopline_statuses)>0:
                    for stopline_id, stopline_linestring in self.all_stoplines.items():
                        if stopline_id in stopline_statuses and stopline_statuses[stopline_id]==0 and local_path_buffer.intersects(stopline_linestring):
                            stopline_centroid = shapely.centroid(stopline_linestring)
                            stopline_point = np.array([(
                            stopline_centroid.x,
                            stopline_centroid.y,
                            0.0,
                            0.0, # vx
                            0.0, # vy
                            0.0, # vz
                            self.braking_safety_distance_stopline,
                            np.inf,
                            2 # Category 2
                            )], dtype=DTYPE)
                            collision_points = np.append(collision_points, stopline_point)

                for obj in detected_objects:
                    object_hull = shapely.polygons(np.array(obj.convex_hull).reshape(-1, 3))

                    if local_path_buffer.intersects(object_hull):
                        intersection = local_path_buffer.intersection(object_hull)
                        intersection_points = shapely.get_coordinates(intersection)
                        object_speed = math.sqrt(obj.velocity.x**2 + obj.velocity.y**2 + obj.velocity.z**2) #obj.velocity - geometry_msgs/Vector3 

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

    def traffic_light_status_callback(self, msg):
        stopline_statuses = {}
        for result in msg.results:
            stopline_statuses[result.stopline_id] = result.recognition_result

        self.stopline_statuses = stopline_statuses

        return
    
    def run(self):
        rospy.spin()

def get_stoplines(lanelet2_map):
    """
    Add all stop lines to a dictionary with stop_line id as key and stop_line as value
    :param lanelet2_map: lanelet2 mapdef 
    """

    stoplines = {}
    for line in lanelet2_map.lineStringLayer:
        if line.attributes:
            if line.attributes["type"] == "stop_line":
                # add stopline to dictionary and convert it to shapely LineString
                stoplines[line.id] = LineString([(p.x, p.y) for p in line])

    return stoplines

if __name__ == '__main__':
    rospy.init_node('collision_points_manager')
    node = CollisionPointsManager()
    node.run()