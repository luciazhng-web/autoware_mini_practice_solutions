#!/usr/bin/env python3

import rospy
import numpy as np

from shapely.geometry import LineString, Point
from shapely import prepare, distance
from tf.transformations import euler_from_quaternion
from scipy.interpolate import interp1d

from autoware_mini.msg import Path
from geometry_msgs.msg import PoseStamped
from autoware_mini.msg import VehicleCmd

class PurePursuitFollower:
    def __init__(self):

        # Parameters
        self.lookahead_distance = rospy.get_param('~lookahead_distance')
        self.wheel_base = rospy.get_param("/vehicle/wheel_base") #for steer angle calculation

        # initialization
        self.path = None
        self.current_pose = None
        self.path_linestring = None
        self.distance_to_velocity_interpolator = None

        # Publishers
        self.vehicle_cmd_pub = rospy.Publisher('/control/vehicle_cmd', VehicleCmd, queue_size=1)

        # Subscribers
        rospy.Subscriber('path', Path, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def path_callback(self, msg):
        # receives local path to be followed
        self.path = msg

        # convert waypoints to shapely linestring
        self.path_linestring = LineString([(w.position.x, w.position.y) for w in msg.waypoints])
        # prepare path - creates spatial tree, making the spatial queries more efficient
        prepare(self.path_linestring)

        # Create a distance-to-velocity interpolator for the path

        # collect waypoint x and y coordinates
        waypoints_xy = np.array([(w.position.x, w.position.y) for w in msg.waypoints])
        # Calculate distances between points
        distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xy, axis=0)**2, axis=1)))
        # add 0 distance in the beginning
        distances = np.insert(distances, 0, 0)
        # Extract velocity values at waypoints
        velocities = np.array([w.speed for w in msg.waypoints])

        # create interpolator
        try:
            self.distance_to_velocity_interpolator = interp1d(
                distances, 
                velocities, 
                kind='linear', 
                bounds_error=False, 
                fill_value=0.0
            )
            rospy.loginfo(f"Received a new path with {len(self.path.waypoints)} waypoints. Velocity interpolator created.")
        except ValueError as e:
            rospy.logerr(f"Could not create interpolator: {e}.")
            self.distance_to_velocity_interpolator = None 

    def current_pose_callback(self, msg):
        # receives current pose
        self.current_pose = msg
        
        if self.path_linestring is None or self.distance_to_velocity_interpolator is None:
            return 

        #x = msg.pose.position.x
        #y = msg.pose.position.y
        #z = msg.pose.position.z
        #orientation = msg.pose.orientation

        #rospy.loginfo(f'/bicycle_simulation- initial position ({x},{y},{z}) orientation ({orientation}) in map frame.')

        # current position and projected position to path(trajectory)
        current_pose = Point([msg.pose.position.x, msg.pose.position.y])
        d_ego_from_path_start = self.path_linestring.project(current_pose)

        # using euler_from_quaternion to get the heading angle
        _, _, heading = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]) #map reference

        # lookahead point
        d_ego_to_point = d_ego_from_path_start + self.lookahead_distance
        lookahead_point = self.path_linestring.interpolate(d_ego_to_point)
        ld = distance(current_pose, lookahead_point)

        # lookahead point heading calculation
        lookahead_heading = np.arctan2(lookahead_point.y - current_pose.y, lookahead_point.x - current_pose.x) #map reference
        alpha = lookahead_heading - heading
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
                
        # calculate steer angle
        steering_angle = np.arctan2(2*self.wheel_base*np.sin(alpha), ld)

        # calcuate velocity
        velocity = self.distance_to_velocity_interpolator(d_ego_from_path_start).item()
        velocity = max(0.0, velocity)

        # define vehicle command
        vehicle_cmd = VehicleCmd()
        
        vehicle_cmd.header.stamp = msg.header.stamp 
        vehicle_cmd.header.frame_id = "base_link" 

        vehicle_cmd.ctrl_cmd.linear_velocity = velocity
        vehicle_cmd.ctrl_cmd.steering_angle = steering_angle

        # Publish command
        self.vehicle_cmd_pub.publish(vehicle_cmd)
        
        rospy.loginfo(f'Ego Distance from Path Start: {d_ego_from_path_start:.2f} meters, with speed: {velocity: .2f} and steering_angle: {steering_angle: .2f}')

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        rospy.init_node('pure_pursuit_follower')
        node = PurePursuitFollower()
        node.run()
    except rospy.ROSInterruptException:
        rospy.logwarn("PurePursuitFollower node interrupted.")