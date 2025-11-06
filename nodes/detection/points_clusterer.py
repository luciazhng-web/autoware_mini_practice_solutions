#!/usr/bin/env python3

import math
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from ros_numpy import numpify,msgify
from numpy.lib.recfunctions import structured_to_unstructured,unstructured_to_structured
from sklearn.cluster import DBSCAN

class PointsCluster:

    def __init__(self):

        #parameter
        self.cluster_epsilon = rospy.get_param('~cluster_epsilon')
        self.cluster_min_size = rospy.get_param('~cluster_min_size')

        self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_size, algorithm='auto')
        
        #publisher
        self.cluster_pub = rospy.Publisher('points_clustered', PointCloud2, queue_size=1, tcp_nodelay=True)
        
        #subscriber
        rospy.Subscriber('points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)
      
    def points_callback(self,msg):
        data = numpify(msg)

        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32) #DBSCAN only works with unstructured array
        labels = self.clusterer.fit_predict(points)

        # filtering the noise
        mask = (labels != -1)
        labels_filtered = labels[mask].reshape(-1, 1) # from (N,) to (N,1)
        clustered_points = points[mask]

        # combine points (N, 3) with labels (N, 1) into one (N, 4) array
        points_labeled = np.concatenate((clustered_points, labels_filtered), axis=1)

        if clustered_points.shape[0] != labels_filtered.shape[0]:
            raise AssertionError(f"Error: {clustered_points.shape[0]}) does not match number of labels ({labels_filtered.shape[0]}")

        # convert labelled points to PointCloud2 format
        data = unstructured_to_structured(points_labeled, dtype=np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('label', np.int32)
        ]))

        # publish clustered points message
        cluster_msg = msgify(PointCloud2, data)
        cluster_msg.header.stamp = msg.header.stamp
        cluster_msg.header.frame_id = msg.header.frame_id
        self.cluster_pub.publish(cluster_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('points_clusterer', log_level=rospy.INFO)
    node = PointsCluster()
    node.run()

    
