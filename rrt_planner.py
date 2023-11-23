#!/usr/bin/python
import sys
import time
import pickle
import numpy as np
import random
import cv2
from PIL import Image
from numpy import asarray

from itertools import product
from math import cos, sin, pi, sqrt

from plotting_utils import draw_plan
from priority_queue import priority_dict


class State(object):
    """
    2D state.
    """
   
    def __init__(self, x, y, parent):
        """
        x represents the columns on the image and y represents the rows,
        Both are presumed to be integers
        """
        self.x = x
        self.y = y
        self.parent = parent
        self.children = []

       
    def __eq__(self, state):
         
        return state and self.x == state.x and self.y == state.y

    def __hash__(self):
       
        return hash((self.x, self.y))
   
    def euclidean_distance(self, state):
        assert (state)
        return sqrt((state.x - self.x)**2 + (state.y - self.y)**2)
   
class RRTPlanner(object):
    """
    Applies the RRT algorithm on a given grid world
    """
   
    def __init__(self, world):
        self.world = world
        self.max_x = world.shape[1]
        self.max_y = world.shape[0]

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world[:,:,0]
        self.occ_grid = (self.occ_grid == 0).astype('uint8')
       
    def state_is_free(self, state):
        """
        Does collision detection. Returns true iff the state and its nearby
        surroundings are free.
        """
        free = (self.occ_grid[state.y-2:state.y+2, state.x-2:state.x+2] == 0).all()
        return free

    def sample_state(self):
        """
        Sample a new state uniformly randomly on the image.
        """
        x = int(random.uniform(0,self.max_x-1))
        y = int(random.uniform(0,self.max_y-1))
        return State(x, y, None)
           
    def _follow_parent_pointers(self, state):
        """
        Returns the path [start_state, ..., destination_state] by following the
        parent pointers.
        """
       
        curr_ptr = state
        path = [state]
       
        while curr_ptr is not None:
            path.append(curr_ptr)
            curr_ptr = curr_ptr.parent

        return path[::-1]


    def find_closest_state(self, tree_nodes, state):
        min_dist = float("Inf")
        closest_state = None
        for node in tree_nodes:
            dist = node.euclidean_distance(state)  
            if dist < min_dist:
                closest_state = node
                min_dist = dist

        return closest_state

    def steer_towards(self, s_nearest, s_rand, max_radius):
        """
        Returns a new state s_new whose coordinates x and y
        are decided as follows:
       
        If s_rand is within a circle of max_radius from s_nearest
        then s_new.x = s_rand.x and s_new.y = s_rand.y
       
        Otherwise, s_rand is farther than max_radius from s_nearest.
        In this case we place s_new on the line from s_nearest to
        s_rand, at a distance of max_radius away from s_nearest.
       
        """
       
        if (s_nearest.euclidean_distance(s_rand) <= max_radius) :
            x = int(min(self.max_x-1, s_rand.x))
            y = int(min(self.max_y-1, s_rand.y))
        else :
        # We need to compute the angle theta between the x-axis and the line
        # joining s_nearest and s_rand (s_new is on that line)

            theta = np.arctan2((s_rand.y - s_nearest.y), (s_rand.x - s_nearest.x))

            x = int(min(self.max_x-1, (s_nearest.x + cos(theta) * max_radius)))
            y = int(min(self.max_y-1, (s_nearest.y + sin(theta) * max_radius)))

        s_new = State(x, y, s_nearest)
        return s_new


    def path_is_obstacle_free(self, s_from, s_to):
        """
        Returns true iff the line path from s_from to s_to
        is free
        """
        assert (self.state_is_free(s_from))
       
        if not (self.state_is_free(s_to)):
            return False

        max_checks = 10
        for i in range(max_checks):
            dist_x = s_to.x - s_from.x
            x = int(s_from.x + (float(i)/max_checks)*dist_x)
            dist_y = s_to.y - s_from.y
            y = int(s_from.y + (float(i)/max_checks)*dist_y)
            if not (self.state_is_free(State(x,y,None))) :
                return False      

        return True
   
    def plan(self, start_state, dest_state, max_num_steps, max_steering_radius, dest_reached_radius, live_view=True):
        """
        Returns a path as a sequence of states [start_state, ..., dest_state]
        if dest_state is reachable from start_state. Otherwise returns [start_state].
        Assume both source and destination are in free space.
        """
        assert (self.state_is_free(start_state))
        assert (self.state_is_free(dest_state))

        tree_nodes = set()
        tree_nodes.add(start_state)
       
        img = np.copy(self.world)
        cv2.circle(img, (start_state.x, start_state.y), 2, (255,0,0))

        plan = [start_state]
       
        for step in range(max_num_steps):

            s_rand = self.sample_state()
            s_nearest = self.find_closest_state(tree_nodes, s_rand)
            s_new = self.steer_towards(s_nearest, s_rand, max_steering_radius)
            s_new.parent = s_nearest
           
            if self.path_is_obstacle_free(s_nearest, s_new):
                tree_nodes.add(s_new)
                s_nearest.children.append(s_new)

                if s_new.euclidean_distance(dest_state) < dest_reached_radius:
                    dest_state.parent = s_new
                    plan = self._follow_parent_pointers(dest_state)
                    break
               
                # plot the new node and edge
                cv2.circle(img, (s_new.x, s_new.y), 3, (0,0,255),3)
                cv2.line(img, (s_nearest.x, s_nearest.y), (s_new.x, s_new.y), (255,0,0),2)

            # Keep showing the image for a bit even
            # if we don't add a new node and edge
            if live_view:
                cv2.imshow('image', img)
                cv2.waitKey(100)

        if live_view:
            draw_plan(img, plan, bgr=(0,0,255), thickness=2, show_live=True)
            cv2.waitKey(0)
        return plan
   
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: rrt_planner.py map_image_filename")
        sys.exit(1)

    # load the image
    image = Image.open(sys.argv[1])
    # convert image to numpy array
    world = asarray(image)
    rrt = RRTPlanner(world)

    start_state = State(500, 300, None)
    dest_state = State(10, 10, None)

    max_num_steps = 1000     # max number of nodes to be added to the tree
    max_steering_radius = 30 # pixels
    dest_reached_radius = 50 # pixels
    plan = rrt.plan(start_state,
                    dest_state,
                    max_num_steps,
                    max_steering_radius,
                    dest_reached_radius,
                    live_view=True)
   
    print('RRT planning complete. Saving image.')
    draw_plan(world, plan, bgr=(0,0,255), thickness=2, show_live=False,filename='rrt_result.png')
