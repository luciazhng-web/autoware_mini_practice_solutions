#!/usr/bin/env python3

import math
import rospy

from tf.transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster
from pyproj import CRS, Transformer, Proj

from novatel_oem7_msgs.msg import INSPVA
from geometry_msgs.msg import PoseStamped, TwistStamped, Quaternion, TransformStamped

# convert azimuth to yaw angle
def convert_azimuth_to_yaw(azimuth):
    """
    Converts azimuth to yaw. Azimuth is CW angle from the north. Yaw is CCW angle from the East.
    :param azimuth: azimuth in radians
    :return: yaw in radians
    """
    yaw = -azimuth + math.pi/2
    # Clamp within 0 to 2 pi
    if yaw > 2 * math.pi:
        yaw = yaw - 2 * math.pi
    elif yaw < 0:
        yaw += 2 * math.pi

    return yaw

class Localizer:
    def __init__(self):

        # Parameters
        self.undulation = rospy.get_param('undulation')
        utm_origin_lat = rospy.get_param('utm_origin_lat')
        utm_origin_lon = rospy.get_param('utm_origin_lon')

        # Internal variables
        self.crs_wgs84 = CRS.from_epsg(4326)
        self.crs_utm = CRS.from_epsg(25835)
        self.utm_projection = Proj(self.crs_utm) # By default input is in lon, lat; else should use Transformer
        
        # create a coordinate transformer
        self.transformer = Transformer.from_crs(self.crs_wgs84, self.crs_utm) #if always_xy=True; then in lon, lat order; else in lat, lon order
        self.origin_x, self.origin_y = self.transformer.transform(utm_origin_lat, utm_origin_lon)

        # Subscribers
        rospy.Subscriber('/novatel/oem7/inspva', INSPVA, self.transform_coordinates)

        # Publishers
        self.current_pose_pub = rospy.Publisher('current_pose', PoseStamped, queue_size=10)
        self.current_velocity_pub = rospy.Publisher('current_velocity', TwistStamped, queue_size=10)
        self.br = TransformBroadcaster()

    def transform_coordinates(self, msg):

        # calculate position
        utm_x, utm_y = self.transformer.transform(msg.latitude, msg.longitude)
        p_x = utm_x - self.origin_x
        p_y = utm_y - self.origin_y
        p_z = msg.height - self.undulation
        
        # calculate azimuth correction
        azimuth_correction_degree = self.utm_projection.get_factors(msg.longitude, msg.latitude).meridian_convergence # True north - grid north
        azimuth_correction = math.radians(azimuth_correction_degree) #convert to radians
        azimuth_radians = math.radians(msg.azimuth)
        corrected_azimuth = azimuth_radians - azimuth_correction
        yaw = convert_azimuth_to_yaw(corrected_azimuth)
        
        # convert yaw to quaternion
        x, y, z, w = quaternion_from_euler(0, 0, yaw)
        orientation = Quaternion(x, y, z, w)
        
        # calculate the velocity
        vel_norm = math.sqrt(msg.north_velocity**2 + msg.east_velocity**2)

        # publish current pose
        current_pose_msg = PoseStamped()
        current_pose_msg.header.stamp = msg.header.stamp
        current_pose_msg.header.frame_id = "map"
        current_pose_msg.pose.position.x = p_x
        current_pose_msg.pose.position.y = p_y
        current_pose_msg.pose.position.z = p_z
        current_pose_msg.pose.orientation = orientation
        self.current_pose_pub.publish(current_pose_msg)
	
        # publish current velocity
        current_vel_msg = TwistStamped()
        current_vel_msg.header.stamp = msg.header.stamp
        current_vel_msg.header.frame_id = "base_link"
        current_vel_msg.twist.linear.x = vel_norm
        self.current_velocity_pub.publish(current_vel_msg)

        # create a transform message
        t = TransformStamped()

        # fill in the transform message - t
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "map"
        t.child_frame_id =  "base_link"
        t.transform.translation.x = p_x
        t.transform.translation.y = p_y
        t.transform.translation.z = p_z
        t.transform.rotation = orientation

        # publish transform
        self.br.sendTransform(t)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('localizer')
    node = Localizer()
    node.run()
