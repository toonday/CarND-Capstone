#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 40 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1
        self.debug_str = "."
        self.last_idx = None

        #rospy.logwarn("WaypointUpdater ................")
        self.loop() # rospy.spin()

    def loop(self):
        rate = rospy.Rate(3)
        while not rospy.is_shutdown():
            #rospy.logwarn("WaypointUpdater::Update() ................")
            if self.pose and self.base_waypoints:
                # Get closest waypoint
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        
        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]
        
        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])
        
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        
        if val > 0:
            closest_idx = (closest_idx+1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self, closest_idx):
        wp_length = len(self.base_waypoints.waypoints)
        #farthest_idx = (closest_idx + LOOKAHEAD_WPS) % wp_length
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        
        lane = Lane()
        lane.header = self.base_waypoints.header

        #base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]
        #'''
        farthest_idx_mod = farthest_idx % wp_length
        if farthest_idx_mod <= closest_idx:
        #if farthest_idx < closest_idx:
            base_waypoints = self.base_waypoints.waypoints[closest_idx:wp_length] + self.base_waypoints.waypoints[0:farthest_idx_mod]
        else:
            base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]
        #'''
        
        if not self.last_idx or self.last_idx != self.stopline_wp_idx:
            self.last_idx = self.stopline_wp_idx
            rospy.logwarn("self.stopline_wp_idx: {0}".format(self.stopline_wp_idx))
        #if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx and farthest_idx_mod == farthest_idx):
        if self.stopline_wp_idx == -1 or self.stopline_wp_idx >= farthest_idx:
            lane.waypoints = base_waypoints
            #rospy.logwarn("...")
        else:
            #rospy.logwarn("slowing down waypoints")
            #lane.waypoints = base_waypoints
            lane.waypoints = self.slowdown_waypoints(base_waypoints, closest_idx)
            
        if len(lane.waypoints) != LOOKAHEAD_WPS:
            rospy.logwarn("len of lane.waypoints == {0}".format(len(lane.waypoints)))
        self.final_waypoints_pub.publish(lane)

    def slowdown_waypoints(self, waypoints, closest_idx):
        slowdown_wp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            
            stop_idx_diff = max(self.stopline_wp_idx-(closest_idx+4), 0)
            dist = self.distance(waypoints, i, stop_idx_diff)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.0
                
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            slowdown_wp.append(p)
            
        return slowdown_wp

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg
        #rospy.logwarn("WaypointUpdater::pose_cb() ................")
        #pass

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        #rospy.logwarn("WaypointUpdater::waypoints_cb() ................")
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
        #pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data
        #rospy.logwarn("self.stopline_wp_idx = {0}".format(self.stopline_wp_idx))
        #pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
