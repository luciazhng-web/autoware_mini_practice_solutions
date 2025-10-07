#!/usr/bin/env python

import rospy
from std_msgs.msg import String

rospy.init_node('publisher')

freq = rospy.get_param('~rate', 5) #add for 7 - Reusing nodes
message = rospy.get_param('~message', 'Hello World!') #add for 6 - Parameters in launch files and nodes
rate = rospy.Rate(freq)
pub = rospy.Publisher('/message', String, queue_size=10)


while not rospy.is_shutdown():
    pub.publish(message) #replacing the hardcoded message, for 6 - Parameters in launch files and nodes
    rate.sleep()
