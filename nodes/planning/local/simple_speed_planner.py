#!/usr/bin/env python3

import rospy
import math
import message_filters
import traceback
import shapely
import numpy as np
import threading
from numpy.lib.recfunctions import structured_to_unstructured
from ros_numpy import numpify
from autoware_mini.msg import Path, Log
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from autoware_mini.geometry import project_vector_to_heading, get_distance_between_two_points_2d
from autoware_mini.path import PathWrapper

class SpeedPlanner:

    def __init__(self):

        # parameters
        self.default_deceleration = rospy.get_param("default_deceleration")
        self.braking_reaction_time = rospy.get_param("braking_reaction_time")
        synchronization_queue_size = rospy.get_param("~synchronization_queue_size")
        synchronization_slop = rospy.get_param("~synchronization_slop")
        self.distance_to_car_front = rospy.get_param("distance_to_car_front")

        # variables
        self.collision_points = None
        self.current_position = None
        self.current_speed = None

        # Lock for thread safety
        self.lock = threading.Lock()

        # publishers
        self.local_path_pub = rospy.Publisher('local_path', Path, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)

        collision_points_sub = message_filters.Subscriber('collision_points', PointCloud2, tcp_nodelay=True)
        local_path_sub = message_filters.Subscriber('extracted_local_path', Path, tcp_nodelay=True)

        ts = message_filters.ApproximateTimeSynchronizer([collision_points_sub, local_path_sub], queue_size=synchronization_queue_size, slop=synchronization_slop)

        ts.registerCallback(self.collision_points_and_path_callback)

    def current_velocity_callback(self, msg):
        self.current_speed = msg.twist.linear.x

    def current_pose_callback(self, msg):
        self.current_position = shapely.Point(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def collision_points_and_path_callback(self, collision_points_msg, local_path_msg):
        try:
            with self.lock:
                collision_points = numpify(collision_points_msg) if len(collision_points_msg.data) > 0 else np.array([])
                current_position = self.current_position
                current_speed = self.current_speed

            if current_speed is None or current_position is None:
                rospy.logwarn_throttle(3, "%s - current speed or position not received!", rospy.get_name())
                return

            if  len(collision_points) == 0:
                self.local_path_pub.publish(local_path_msg)
                return
            
            closest_object_distance = 0.0
            closest_object_velocity = 0.0
            stopping_point_distance = 0.0

            # extract collision point distance on local path
            local_path = PathWrapper(local_path_msg.waypoints)
            local_path_linestring = shapely.LineString([(w.position.x, w.position.y) for w in local_path_msg.waypoints])
            collision_points_shapely = shapely.points(structured_to_unstructured(collision_points[['x', 'y', 'z']]))
            collision_point_distances = np.array([local_path_linestring.project(collision_point_shapely) for collision_point_shapely in collision_points_shapely])
            
            ego_distance_localpath = local_path_linestring.project(current_position) # it's uncessary alway 0

            # get distance_to_stop per collision obstacle
            distance_to_stop = collision_points['distance_to_stop']

            # calculate target velocity for every collision point
            object_speed = np.sqrt(collision_points['vx']**2 + collision_points['vy']**2 + collision_points['vz']**2)
            collision_point_path_headings = [local_path.get_heading_at_distance(d) for d in collision_point_distances] #point before and after
            collision_point_velocities = np.array([project_vector_to_heading(heading, Vector3(vx, vy, vz))
                                        for heading, (vx, vy, vz) in zip(collision_point_path_headings, collision_points[['vx', 'vy', 'vz']])])

            # calculate the safe following distance
            collision_point_velocities = np.maximum(0, collision_point_velocities)
            safe_distances = self.braking_reaction_time *abs(collision_point_velocities)

            # calculate deceleration distance
            deceleration_distances = np.maximum(0,collision_point_distances - self.distance_to_car_front - distance_to_stop - safe_distances)

            # calculate target speed
            deceleration_speeds = np.sqrt(np.maximum(0, collision_point_velocities**2 + 2*self.default_deceleration * deceleration_distances))

            # obtain the lowest target velocity
            target_velocity = min(deceleration_speeds)
            min_value_index = np.argmin(deceleration_speeds)
            closest_object_distance = deceleration_distances[min_value_index] + self.distance_to_car_front
            stopping_point_distance = safe_distances[min_value_index] + deceleration_distances[min_value_index] + self.distance_to_car_front
            collision_point_category = collision_points[min_value_index]["category"]
            closest_object_velocity = collision_point_velocities[min_value_index]

            for i, wp in enumerate(local_path_msg.waypoints):
                wp.speed = min(target_velocity, wp.speed)

            # Update the lane message with the calculated values
            path = Path()
            path.header = local_path_msg.header
            path.waypoints = local_path_msg.waypoints
            path.closest_object_distance = closest_object_distance # Distance to the collision point with lowest target velocity (also closest object for now)
            path.closest_object_velocity = closest_object_velocity # Velocity of the collision point with lowest target velocity (0)
            path.is_blocked = True
            path.stopping_point_distance = stopping_point_distance # Stopping point distance can be set to the distance to the closest object for now
            path.collision_point_category = collision_point_category # Category of collision point with lowest target velocity
            self.local_path_pub.publish(path)

        except Exception as e:
            rospy.logerr_throttle(10, "%s - Exception in callback: %s", rospy.get_name(), traceback.format_exc())


    def get_heading_at_distance(self, linestring, distance):
        """
        Get heading of the path at a given distance
        :param distance: distance along the path
        :param linestring: shapely linestring
        :return: heading angle in radians
        """

        point_after_object = linestring.interpolate(distance + 0.1)
        # if distance is negative it is measured from the end of the linestring in reverse direction
        point_before_object = linestring.interpolate(max(0, distance - 0.1))

        # get heading between two points
        return math.atan2(point_after_object.y - point_before_object.y, point_after_object.x - point_before_object.x)


    def project_vector_to_heading(self, heading_angle, vector):
        """
        Project vector to heading
        :param heading_angle: heading angle in radians
        :param vector: vector
        :return: projected vector
        """

        return vector.x * math.cos(heading_angle) + vector.y * math.sin(heading_angle)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('speed_planner')
    node = SpeedPlanner()
    node.run()