#!/usr/bin/env python3

import math
import itertools
import shapely
import rospy
import lanelet2
import copy

from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest

from geometry_msgs.msg import PoseStamped, Pose
from autoware_mini.lanelet2 import load_lanelet2_map
from autoware_mini.msg import Waypoint, Path
from shapely.geometry import LineString, Point
from shapely.ops import substring

class Lanelet2GlobalPlanner:

    def __init__(self):

        # parameter
        lanelet2_map_path = rospy.get_param('~lanelet2_map_path')
        self.speed_limit = rospy.get_param('~speed_limit')
        self.output_frame = rospy.get_param('/planning/output_frame', 'map')
        self.distance_to_goal_limit = rospy.get_param('/planning/distance_to_goal_limit', 5.0)

        # initialization
        self.goal_point = None
        self.current_location = None
        self.lanelet2_map = load_lanelet2_map(lanelet2_map_path)

        # traffic rules
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                          lanelet2.traffic_rules.Participants.VehicleTaxi)
        # routing graph
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)

        # Subscriber
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

        # publisher
        self.waypoints_pub = rospy.Publisher('global_path', Path, queue_size=1, latch=True)
        
    # call back functions
    def goal_callback(self, msg):

        self.goal_pose = msg

        # loginfo message about receiving the goal point
        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                    msg.pose.orientation.w, msg.header.frame_id)
       
        self.current_location = BasicPoint2d(self.current_pose.pose.position.x, self.current_pose.pose.position.y)
        self.goal_point = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        # get start and end lanelets
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]
        
        # find routing graph
        route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)
        
        if route is None:
            rospy.logwarn("%s - No route found between start and goal lanelets.", rospy.get_name())
            return

        # find shortest path
        path = route.shortestPath()
        
        # This returns LaneletSequence to a point where a lane change would be necessary to continue
        path_no_lane_change = path.getRemainingLane(start_lanelet)

        rospy.loginfo(f"path_no_lane_change: {path_no_lane_change}")

        waypoints = self.convert_to_waypoints(path_no_lane_change)

        if waypoints is None:
            rospy.logerr("%s - route contained an impossible lane change!", rospy.get_name())
            return
            
        # Convert waypoints to Shapely LineString
        full_path = LineString([(w.position.x, w.position.y) for w in waypoints])

        # Convert both current pose and goal point to Shapely Points
        start_p = shapely.Point(self.current_pose.pose.position.x, self.current_pose.pose.position.y)
        goal_p = shapely.Point(msg.pose.position.x, msg.pose.position.y)
        
        # Project the start point and goal point to the global path
        start_dis = full_path.project(start_p)
        goal_dis = full_path.project(goal_p)
        start_on_path = full_path.interpolate(start_dis)
        goal_on_path = full_path.interpolate(goal_dis)
        
        # Get the updated path
        selected_waypoints = []
        for w in waypoints:
            wp_dis = full_path.project(Point(w.position.x, w.position.y))
            if start_dis <= wp_dis <= goal_dis:
                selected_waypoints.append(w)
        
        # Update the last waypoint
        last_wp = selected_waypoints[-1]
        new_wp = copy.deepcopy(last_wp)
        new_wp.position.x = goal_on_path.x
        new_wp.position.y = goal_on_path.y
        selected_waypoints.append(new_wp)

        self.publish_waypoints(selected_waypoints)

    def convert_to_waypoints(self, lanelet_sequence):

        waypoints = []
        last_lanelet = False
        
        for i, lanelet in enumerate(lanelet_sequence):
            if i == len(lanelet_sequence)-1:
                last_lanelet = True

            # Fetch speed from lanelet attributes
            speed = self.speed_limit / 3.6
            if 'speed_limit' in lanelet.attributes:
                speed = min(speed, float(lanelet.attributes['speed_limit']) / 3.6)
            if 'speed_ref' in lanelet.attributes:
                speed = min(speed, float(lanelet.attributes['speed_ref']) / 3.6)

            for idx, point in enumerate(lanelet.centerline):
                if not last_lanelet and idx == len(lanelet.centerline)-1:
                    break

                # create Waypoint (from autoware_mini.msgs import Waypoint) and get the coordinats from lanelet.centerline points
                waypoint = Waypoint()
                waypoint.position.x = point.x
                waypoint.position.y = point.y
                waypoint.position.z = point.z
                waypoint.speed = speed

                waypoints.append(waypoint)

        return waypoints

    def publish_waypoints(self, waypoints):

        path = Path()
        path.header.frame_id = self.output_frame
        path.header.stamp = rospy.Time.now()
        path.waypoints = waypoints
        self.waypoints_pub.publish(path)

    def current_pose_callback(self, msg):
        self.current_pose = msg
       
        if self.goal_point != None:
            dx = self.current_pose.pose.position.x - self.goal_pose.pose.position.x
            dy = self.current_pose.pose.position.y - self.goal_pose.pose.position.y
            d = math.sqrt(dx**2+dy**2)
            if d < self.distance_to_goal_limit:
                self.goal_point = None
                self.lanelet_candidates = []
                self.publish_waypoints([])
                rospy.loginfo("%s - goal has been reached, clearing path!", rospy.get_name())

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner', log_level=rospy.INFO)
    node = Lanelet2GlobalPlanner()
    node.run()

