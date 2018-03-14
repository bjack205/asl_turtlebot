import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rospy
import time
import random
import heapq


# Represents a motion planning problem to be solved using A*
class AStar(object):

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., (-5, -5))
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., (5, 5))
        self.occupancy = occupancy                 # occupancy grid
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()     # the set containing the states that are condidate for future expension

        self.f_score = {}       # dictionary of the f score (estimated cost from start to goal passing through state)
        self.g_score = {}       # dictionary of the g score (cost-to-go from start to state)
        self.came_from = {}     # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(x_init)
        self.g_score[x_init] = 0
        self.f_score[x_init] = self.distance(x_init,x_goal)
        
        self.neigh_end_time = 0
        self.coll_end_time = 0

        self.path = None        # the final path as a list of states

    # Checks if a give state is free, meaning it is inside the bounds of the map and
    # is not inside any obstacle
    # INPUT: (x)
    #          x - tuple state
    # OUTPUT: Boolean True/False
    def is_free(self, x):
        if x==self.x_init or x==self.x_goal:
            return True
        for dim in range(len(x)):
            if x[dim] < self.statespace_lo[dim]:
                return False
            if x[dim] >= self.statespace_hi[dim]:
                return False
        
        start = time.time()
        if not self.occupancy.is_free(x):
            self.coll_end_time = self.coll_end_time + time.time() - start
            return False
            
        self.coll_end_time = self.coll_end_time + time.time() - start
        return True

    # computes the euclidean distance between two states
    # INPUT: (x1, x2)
    #          x1 - first state tuple
    #          x2 - second state tuple
    # OUTPUT: Float euclidean distance
    def distance(self, x1, x2):
        return np.linalg.norm(np.array(x1)-np.array(x2))

    # returns the closest point on a discrete state grid
    # INPUT: (x)
    #          x - tuple state
    # OUTPUT: A tuple that represents the closest point to x on the discrete state grid
    def snap_to_grid(self, x):
        #print x, self.resolution, (round(self.resolution*round(x[0]/self.resolution),1), round(self.resolution*round(x[1]/self.resolution)),1), round(x[1]/self.resolution)
        return (self.resolution*round(x[0]/self.resolution), self.resolution*round(x[1]/self.resolution))

    # gets the FREE neighbor states of a given state. Assumes a motion model
    # where we can move up, down, left, right, or along the diagonals by an
    # amount equal to self.resolution.
    # Use self.is_free in order to check if any given state is indeed free.
    # Use self.snap_to_grid (see above) to ensure that the neighbors you compute
    # are actually on the discrete grid, i.e., if you were to compute neighbors by
    # simply adding/subtracting self.resolution from x, numerical error could
    # creep in over the course of many additions and cause grid point equality
    # checks to fail. To remedy this, you should make sure that every neighbor is
    # snapped to the grid as it is computed.
    # INPUT: (x)
    #           x - tuple state
    # OUTPUT: List of neighbors that are free, as a list of TUPLES
    def get_neighbors(self, x):
        # TODO: fill me in!
        neighbors = []
        start = time.time()
        # move right
        neigh_x = x[0] + self.resolution
        neigh_y = x[1]
        neigh = (neigh_x , neigh_y)
        if self.is_free(neigh):
            neighbors.append(self.snap_to_grid(neigh))
        
        # move left
        neigh_x = x[0] - self.resolution
        neigh_y = x[1]
        neigh = (neigh_x, neigh_y)
        if self.is_free(neigh):
            neighbors.append(self.snap_to_grid(neigh))
        
        # move up
        neigh_x = x[0]
        neigh_y = x[1] + self.resolution
        neigh = (neigh_x, neigh_y)
        if self.is_free(neigh):
            neighbors.append(self.snap_to_grid(neigh))
        
        # move down
        neigh_x = x[0]
        neigh_y = x[1] - self.resolution
        neigh = (neigh_x, neigh_y)
        if self.is_free(neigh):
            neighbors.append(self.snap_to_grid(neigh))
        
        """
        # move diagonally
        neigh_x = x[0] + self.resolution/np.sqrt(2)
        neigh_y = x[1] + self.resolution/np.sqrt(2)
        neigh = (neigh_x, neigh_y)
        if self.is_free(neigh):
            neighbors.append(self.snap_to_grid(neigh))
        """
        self.neigh_end_time = self.neigh_end_time + time.time() - start
        random.shuffle(neighbors)
        return neighbors           
    
    # Line of SIght
    def LineofSight(self,s,s_p):
        
        x0 = round(s[0]/self.resolution)
        y0 = round(s[1]/self.resolution)
        x1 = round(s_p[0]/self.resolution)
        y1 = round(s_p[1]/self.resolution)
        dy = y1 - y0
        dx = x1 - x0
        f = 0
        
        if dy < 0:
            dy = -dy
            sy = -1
        else:
            sy = 1
            
        if dx < 0:
            dx = -dx
            sx = -1
        else:
            sx = 1
        
        
        
        if dx >= dy:
            while x0 != x1:
                f = f + dy
                s_query = (x0+((sx-1)/2), y0+((sy-1)/2))
                s_free = self.is_free(s_query)
                if f >= dx:
                    if not s_free:
                        return False
                    y0 = y0 + sy
                    f = f - dx
                
                if f != 0 and not s_free:
                    return False
                    
                
                s_query2 = (x0+((sx-1)/2), y0)
                s_free2 = self.is_free(s_query2)
                s_query3 = (x0+((sx-1)/2), y0-1)
                s_free3 = self.is_free(s_query3)
                if dy == 0 and not s_free2  and not s_free3:
                    return False
                x0 = x0 + sx
                #print x0, x1
        else:
            while y0 != y1:
                f = f + dx
                s_query = (x0+((sx-1)/2), y0+((sy-1)/2))
                s_free = self.is_free(s_query)
                if f >= dy:
                    if not s_free:
                        return False
                    x0 = x0 + sx
                    f = f - dy
                
                if f != 0 and not s_free:
                    return False                    
                 
                s_query2 = (x0, y0+((sy-1)/2))
                s_free2 = self.is_free(s_query2)
                s_query3 = (x0-1, y0+((sy-1)/2))
                s_free3 = self.is_free(s_query3)                   
                if dx == 0 and not s_query2 and not s_query3:
                    return False
                
                y0 = y0 + sy
                #print y0, y1
        return True            

    # Gets the state in open_set that has the lowest f_score
    # INPUT: None
    # OUTPUT: A tuple, the state found in open_set that has the lowest f_score
    def find_best_f_score(self):
        return min(self.open_set, key=lambda x: self.f_score[x])

    # Use the came_from map to reconstruct a path from the initial location
    # to the goal location
    # INPUT: None
    # OUTPUT: A list of tuples, which is a list of the states that go from start to goal
    def reconstruct_path(self):
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    # Plots the path found in self.path and the obstacles
    # INPUT: None
    # OUTPUT: None
    def plot_path(self):
        if not self.path:
            return
            
        fig = plt.figure()

        self.occupancy.plot(fig.number)

        solution_path = np.array(self.path) * self.resolution
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="solution path", zorder=10)
        plt.scatter([self.x_init[0]*self.resolution, self.x_goal[0]*self.resolution], [self.x_init[1]*self.resolution, self.x_goal[1]*self.resolution], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", np.array(self.x_init)*self.resolution + np.array([.2, 0]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal)*self.resolution + np.array([.2, 0]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis('equal')
        plt.show()

    # Solves the planning problem using the A* search algorithm. It places
    # the solution as a list of of tuples (each representing a state) that go
    # from self.x_init to self.x_goal inside the variable self.path
    # INPUT: None
    # OUTPUT: Boolean, True if a solution from x_init to x_goal was found
    def solve(self):
        while len(self.open_set)>0:
            # TODO: fill me in!
            x_current = self.find_best_f_score()
            if (x_current == self.x_goal):
                self.path = self.reconstruct_path()
                print ("neigbors time: %f", self.neigh_end_time)
                print ("collision time: %f", self.coll_end_time)
                return True
                
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)
            
            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                    continue
                    
                tentative_g_score = self.g_score[x_current] + self.distance(x_current,x_neigh)
                self.came_from[x_neigh] = x_current
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                elif tentative_g_score > self.g_score[x_neigh]:
                    continue
                    
                self.came_from[x_neigh] = x_current
                self.g_score[x_neigh] = tentative_g_score
                self.f_score[x_neigh] = tentative_g_score + self.distance(x_neigh, x_goal)                
                
        return False

# A 2D state space grid with a set of rectangular obstacles. The grid is fully deterministic
class DetOccupancyGrid2D(object):
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        for obs in self.obstacles:
            inside = True
            for dim in range(len(x)):
                if x[dim] < obs[0][dim] or x[dim] > obs[1][dim]:
                    inside = False
                    break
            if inside:
                return False
        return True

    def plot(self, fig_num=0):
        fig = plt.figure(fig_num)
        for obs in self.obstacles:
            ax = fig.add_subplot(111, aspect='equal')
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))

### TESTING

# A simple example

width = 500
height = 500
x_init = (0,0)
x_goal = (5,5)
#obstacles = [((6,6),(8,7)),((2,1),(4,2)),((2,4),(4,6)),((6,2),(8,4))]
obstacles = [((109,446),(126,456)),((348,355),(354,374)),((413,14),(420,32)),((398,426),(420,442)),((274,309),(292,331)),((176,161),(194,179)),((226,149),(239,170)),((268,264),(291,280)),((265,483),(279,498)),((50,141),(62,161)),((150,335),(169,358)),((261,213),(280,234)),((326,417),(349,430)),((122,76),(127,94)),((141,465),(162,476)),((495,380),(508,392)),((405,332),(428,356)),((310,405),(327,415)),((130,488),(153,498)),((361,111),(369,121)),((150,268),(158,286)),((316,389),(328,400)),((489,357),(498,363)),((212,432),(235,437)),((402,83),(422,92)),((77,390),(97,414)),((203,424),(220,447)),((91,54),(107,76)),((188,378),(195,393)),((81,171),(98,191)),((64,491),(72,501)),((466,320),(479,344)),((268,147),(274,157)),((464,59),(483,77)),((288,494),(308,509)),((238,340),(253,362)),((182,66),(203,85)),((180,99),(190,112)),((72,72),(84,92)),((368,87),(380,99)),((354,173),(378,190)),((315,348),(330,361)),((484,122),(505,139)),((113,474),(121,484)),((155,189),(164,212)),((205,212),(219,225)),((164,324),(178,331))]
occupancy = DetOccupancyGrid2D(width, height, obstacles)
"""

# A large random example
width = 101
height = 101
num_obs = 15
min_size = 5
max_size = 25
obs_corners_x = np.random.randint(0,width,num_obs)
obs_corners_y = np.random.randint(0,height,num_obs)
obs_lower_corners = np.vstack([obs_corners_x,obs_corners_y]).T
obs_sizes = np.random.randint(min_size,max_size,(num_obs,2))
obs_upper_corners = obs_lower_corners + obs_sizes
obstacles = zip(obs_lower_corners,obs_upper_corners)
occupancy = DetOccupancyGrid2D(width, height, obstacles)
x_init = tuple(np.random.randint(0,width-2,2).tolist())
x_goal = tuple(np.random.randint(0,height-2,2).tolist())
while not (occupancy.is_free(x_init) and occupancy.is_free(x_goal)):
    x_init = tuple(np.random.randint(0,width-2,2).tolist())
    x_goal = tuple(np.random.randint(0,height-2,2).tolist())
    
"""
"""
astar = AStar((0, 0), (width, height), x_init, x_goal, occupancy)

if not astar.solve():
    print "No path found"
    exit(0)
    
astar.plot_path()
"""
