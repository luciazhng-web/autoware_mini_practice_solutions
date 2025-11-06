#!/usr/bin/env python3

import rospy
import numpy as np

from shapely import MultiPoint
from tf2_ros import TransformListener, Buffer, TransformException
from numpy.lib.recfunctions import structured_to_unstructured
from ros_numpy import numpify, msgify

from sensor_msgs.msg import PointCloud2
from autoware_mini.msg import DetectedObjectArray, DetectedObject
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point32


BLUE80P = ColorRGBA(0.0, 0.0, 1.0, 0.8)

class ClusterDetector:
    def __init__(self):
        self.min_cluster_size = rospy.get_param('~min_cluster_size')
        self.output_frame = rospy.get_param('/detection/output_frame')
        self.transform_timeout = rospy.get_param('~transform_timeout')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('points_clustered', PointCloud2, self.cluster_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())


    def cluster_callback(self, msg):
        data = numpify(msg)
        data_us = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        labels = structured_to_unstructured(data[['label']], dtype=np.int32).flatten()
        points = data_us[:, 0:3]
       
        if msg.header.frame_id != self.output_frame:
        # fetch transform for target frame
            try:
                transform = self.tf_buffer.lookup_transform(self.output_frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(self.transform_timeout))
            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return
        
            tf_matrix = numpify(transform.transform).astype(np.float32)

            # turn into homogeneous coordinates
            points_homogeneous = np.ones((points.shape[0], 4), dtype=np.float32)
            points_homogeneous[:, :3] = points

            # transform points to target frame - map
            points_homogeneous = points_homogeneous.dot(tf_matrix.T) #still (N,4)

        # create detected objects
        detected_objects = []
        unique_labels = np.unique(labels)

        for i in unique_labels:
            mask = (labels == i)
            points3d = points_homogeneous[mask, :3]
            points_2d = MultiPoint(points_homogeneous[mask,:2])
            if points3d.shape[0] < self.min_cluster_size:
                continue

            centroid_x, centroid_y, centroid_z = np.mean(points3d, axis=0)

            # create convex hull
            hull = points_2d.convex_hull
            convex_hull_points = [a for hull in [[x, y, centroid_z] for x, y in hull.exterior.coords] for a in hull]

            # create detected object
            detected_object = DetectedObject()
            detected_object.centroid.x = centroid_x
            detected_object.centroid.y = centroid_y
            detected_object.centroid.z = centroid_z
            detected_object.convex_hull = convex_hull_points

            detected_object.id = i
            detected_object.label = "unknown"
            detected_object.color = BLUE80P
            detected_object.valid = True
            detected_object.position_reliable = True
            detected_object.velocity_reliable = False
            detected_object.acceleration_reliable = False

            detected_objects.append(detected_object)


        # pubpish DetectedObjectArray
        objects_array = DetectedObjectArray()
        objects_array.header.stamp = msg.header.stamp
        objects_array.header.frame_id = self.output_frame
        objects_array.objects = detected_objects

        self.objects_pub.publish(objects_array)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('cluster_detector', log_level=rospy.INFO)
    node = ClusterDetector()
    node.run()
