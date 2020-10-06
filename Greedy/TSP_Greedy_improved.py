#!/usr/bin/env python
# coding: utf-8

# In[8]:


# coding: utf-8

# <hr style="border:2px solid gray"> </hr>
# 
# # Homework 1 - Traveling Salesman Problem
# 
# ## Example Code
# 
# ### Algorithm 1: Greedy
# 
# ### Author: Wangduk Seo (CAU AI Lab)
# <hr style="border:2px solid gray"> </hr>

# # Step 0. Importing packages and Global Settings

# In[ ]:


# package list
import tkinter as tk
from tkinter import filedialog
import numpy as np
import sys
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import time

# Global Variables
# Hill Climbing
SUB_ITERATIONS = 2000  # Iteration of 2-opt search in each evaluation
MAX_EVALUATION = 20  # Max hill climbing iterations

# Plot Settings
PLOT_MODE = True  # Draw Route
PLT_INTERVAL = 100  # Draw Route every 100 iterations
plt.ion()

# First City Index
FIRST_IDX = 0


# # Step 1. Data Loading

# In[ ]:


def fileloader():
    # Data loading
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    if file_path == '':
        raise Exception('Cannot load a data file')
    root.destroy()
    #     Data Format
    #     ---------------------------------------------------------
    #     NAME : pia3056
    #     COMMENT : Bonn VLSI data set with 3056 points
    #     COMMENT : Uni Bonn, Research Institute for Discrete Math
    #     COMMENT : Contributed by Andre Rohe
    #     TYPE : TSP
    #     DIMENSION : 3056 -----------------------------|
    #     EDGE_WEIGHT_TYPE : EUC_2D                     |
    #     NODE_COORD_SECTION                            |
    #     1 0 11 (2 dimentional coordinate of city)     |
    #     2 0 115                                       |
    #     ...                                           |
    #     ...(Total 3056 nodes)<------------------------|
    #     EOF
    #     ---------------------------------------------------------
    with open(file_path, "r") as file:
        file_str = file.readlines()

    # Get the coordinates of cities
    coord_str = file_str[8:-1]  # first city string to last city string (EOF 전까지)
    coord_list = np.zeros((len(coord_str), 2))
    for idx, item in enumerate(coord_str):
        coord_split = item.split()
        coord_list[idx, 0] = int(coord_split[1])
        coord_list[idx, 1] = int(coord_split[2])

    return coord_list


# # Step 2. Searching a path
# ## Algorithm 1. Greedy Algorithm

# In[ ]:


def path_cost(path_map, path):
    # The array of cost between cities in the path
    cnt_cities = path_map.shape[0]
    cost_arr = np.zeros(cnt_cities)
    for i in range(cnt_cities):
        cost_arr[i] = path_map[path[i], path[i + 1]]

    return cost_arr


def greedy(coord_list):
    cnt_cities = len(coord_list)
    # Initialize path and insert first city index to the first and last elements
    best_path = np.zeros(cnt_cities + 1, dtype=np.int)
    best_path[0], best_path[-1] = FIRST_IDX, FIRST_IDX

    # Euclidean distance map between cities
    path_map = euclidean_distances(coord_list, coord_list)

    cities_tovisit = np.ones((cnt_cities), dtype=np.bool)
    cities_tovisit[FIRST_IDX] = False

    # Iteratively Connect nearest cities
    for i in range(1, cnt_cities):
        start_idx = best_path[i - 1]
        distance_from_start = path_map[start_idx, :]
        nearest_list = np.argsort(distance_from_start)
        for idx in range(len(nearest_list)):
            # check the nearest city is visited
            if cities_tovisit[nearest_list[idx]]:
                nearest_city = nearest_list[idx]
                break
        cities_tovisit[nearest_city] = False
        best_path[i] = nearest_city

    for i in range(len(best_path) - 2):
        for j in range(len(best_path) - 1):
            if is_cross_pt(int(coord_list[best_path[i], 0]), int(coord_list[best_path[i], 1]),
                           int(coord_list[best_path[i + 1], 0]),
                           int(coord_list[best_path[i + 1], 1]), int(coord_list[best_path[j], 0]),
                           int(coord_list[best_path[j], 1]),
                           int(coord_list[best_path[j + 1], 0]), int(coord_list[best_path[j + 1], 1])):
                if (i == j):
                    continue
                # best_path[i + 1: j + 1] = best_path[i + 1: j + 1].reverse()
                best_path[i + 1: j + 1] = np.flip(best_path[i + 1: j + 1])
                i = 1


    cost_arr = path_cost(path_map, best_path)
    best_cost = cost_arr.sum()

    # Draw Route
    if PLOT_MODE:
        plt.close()
        figure, ax = plt.subplots()
        plt.scatter(coord_list[:, 0], coord_list[:, 1], c='yellow', s=10)
        plt.title('City Route')
        coord_path = coord_list
        coord_path = np.append(coord_path, coord_path[best_path[0], :].reshape(1, 2), axis=0)
        coord_path[:, :] = coord_path[best_path, :]
        lines, = ax.plot(coord_path[:, 0], coord_path[:, 1], 'k--')
        figure.canvas.draw()
        figure.canvas.flush_events()

    return best_path, best_cost


def is_divide_pt(x11, y11, x12, y12, x21, y21, x22, y22):
    """input: 4 points
    output: True/False
    """
    #  // line1 extension이 line2의 두 점을 양분하는지 검사..
    # 직선의 양분 판단
    f1 = (x12 - x11) * (y21 - y11) - (y12 - y11) * (x21 - x11)
    f2 = (x12 - x11) * (y22 - y11) - (y12 - y11) * (x22 - x11)
    if f1 * f2 < 0:
        return True
    else:
        return False


def is_cross_pt(x11, y11, x12, y12, x21, y21, x22, y22):
    b1 = is_divide_pt(x11, y11, x12, y12, x21, y21, x22, y22)
    b2 = is_divide_pt(x21, y21, x22, y22, x11, y11, x12, y12)
    if b1 and b2:
        return True


# # Main

# In[ ]:


# Step 1
try:
    coord_list = fileloader()
except Exception as e:
    print('예외 발생', e)
    sys.exit()

start_time = time.time()

# Step 2
best_path, best_cost = greedy(coord_list)




print('Execution Time: ' + str(time.time() - start_time))
print('Path: ' + str(best_path.tolist()))
print('Cost: ' + str(best_cost))

# In[ ]:


# In[ ]:




