#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:51:00 2020

@author: gabryelsr
"""

import numpy as np
import rospy
from mavros_msgs.msg import GlobalPositionTarget, State, PositionTarget
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32, Float64, String, Float64MultiArray
import time
from pyquaternion import Quaternion
import math
import threading
from mavros_msgs.msg import *
from mavros_msgs.srv import *
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import deepcopy
from matplotlib import animation
import json
from datetime import datetime

plt.rcParams["figure.autolayout"] = True

#Solução###############################################################################################################################
class SplinePath:
    """Class to represent a path as a cubic spline."""

    # Constructor
    def __init__(self, environment, control_points=[], resolution=100):
        self.environment = environment
        self.control_points = control_points
        self.resolution = resolution

    # Create random control points
    @staticmethod
    def random(environment, num_control_points=10, resolution=100):
        control_points = np.random.rand(num_control_points, 2) * np.array([environment.width, environment.height])
        return SplinePath(environment, control_points, resolution)

    # Create control points from list
    @staticmethod
    def from_list(environment, xy, resolution=100, normalized=False):
        control_points = np.array(xy).reshape(-1, 2)
        if normalized:
            control_points[:,0] *= environment.width
            control_points[:,1] *= environment.height

        return SplinePath(environment, control_points, resolution)

    # Get path
    def get_path(self):

        # Add start and goal to control points
        start = self.environment.start
        goal = self.environment.goal
        points = np.vstack((start, self.control_points, goal))

        # Create spline
        t = np.linspace(0, 1, len(points))
        cs = CubicSpline(t, points, bc_type='clamped')

        # Get path
        tt = np.linspace(0, 1, self.resolution)
        path = cs(tt)

        # Clip path to environment
        path = self.environment.clip_path(path)

        return path

#Ambiente##############################################################################################################################
class Environment:
    """Class to represent the environment in which the mobile robot is operating."""

    # Constructor
    def __init__(self, width=100, height=100, robot_radius=0, obstacles=[], start=None, goal=None):
        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        self.obstacles = obstacles
        self.start = start
        self.goal = goal

    # Method to add an obstacle to the environment
    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    # Method to add a list of obstacles to the environment
    def add_obstacles(self, obstacles):
        self.obstacles.extend(obstacles)

    # Method to remove all obstacles from the environment
    def clear_obstacles(self):
        self.obstacles = []

    # Method to check if a point is in collision with an obstacle
    def in_collision(self, point):
        for obstacle in self.obstacles:
            if obstacle.in_collision(point, self.robot_radius):
                return True
        return False

    # Method to check if a path is in collision with an obstacle
    def path_in_collision(self, path):
        for point in path:
            if self.in_collision(point):
                return True
        return False

    # Method to check if a point is in the environment
    def in_environment(self, point):
        min_x = 0 + self.robot_radius
        max_x = self.width - self.robot_radius
        min_y = 0 + self.robot_radius
        max_y = self.height - self.robot_radius
        return (min_x <= point[0] <= max_x) and (min_y <= point[1] <= max_y)

    # Method to check if a path is in the environment
    def path_in_environment(self, path):
        for point in path:
            if not self.in_environment(point):
                return False
        return True

    # Method to clip a point to the environment
    def clip_point(self, point):
        min_x = 0 + self.robot_radius
        max_x = self.width - self.robot_radius
        min_y = 0 + self.robot_radius
        max_y = self.height - self.robot_radius
        return np.array([np.clip(point[0], min_x, max_x), np.clip(point[1], min_y, max_y)])

    # Method to clip a path to the environment
    def clip_path(self, path):
        clipped_path = []
        for point in path:
            clipped_path.append(self.clip_point(point))
        return np.array(clipped_path)

    # Method to check if a point is in the goal region
    def in_goal(self, point):
        return np.linalg.norm(point - self.goal) <= self.robot_radius

    # Method to check if a path is in the goal region
    def path_in_goal(self, path):
        return self.in_goal(path[-1])

    # Method to check if a point is in the start region
    def in_start(self, point):
        return np.linalg.norm(point - self.start) <= self.robot_radius

    # Method to check if a path is in the start region
    def path_in_start(self, path):
        return self.in_start(path[0])

    # Count number of violations
    def count_violations(self, path):

        # Initialization
        violations = 0
        details = {
            'start_violation': False,
            'goal_violation': False,
            'environment_violation': False,
            'environment_violation_count': 0,
            'collision_violation': False,
            'collision_violation_count': 0,
        }

        # Check the start
        if not self.path_in_start(path):
            violations += 1
            details['start_violation'] = True

        # Check the goal
        if not self.path_in_goal(path):
            violations += 1
            details['goal_violation'] = True

        for point in path:
            if not self.in_environment(point):
                violations += 1
                details['environment_violation_count'] += 1

            for obstacle in self.obstacles:
                if obstacle.in_collision(point, self.robot_radius):
                    violations += 1
                    details['collision_violation_count'] += 1

        details['environment_violation'] = details['environment_violation_count'] > 0
        details['collision_violation'] = details['collision_violation_count'] > 0

        return violations, details

    # Method to check if a path is valid
    def path_is_valid(self, path):
        return self.count_violations(path) == 0

    # Compute the length of a path
    def path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i] - path[i + 1])
        return length

class Obstacle:
    """Class to represent an obstacle in the environment."""

    # Constructor
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    # Method to check if a point is in collision with the obstacle
    def in_collision(self, point, robot_radius=0):
        return np.linalg.norm(point - self.center) <= self.radius + robot_radius

class RectangularObstacle:
        """Rotated rectangle obstacle."""
    
        def __init__(self, center, width, height, angle=0):
            self.center = np.array(center)
            self.width = width
            self.height = height
            self.angle = np.deg2rad(angle)  # Convert degrees to radians
    
            # Precompute rotation matrix and inverse
            cos_a = np.cos(self.angle)
            sin_a = np.sin(self.angle)
            self.rot_matrix = np.array([[cos_a, -sin_a],
                                        [sin_a,  cos_a]])
            self.inv_rot_matrix = np.array([[cos_a, sin_a],
                                            [-sin_a, cos_a]])
    
        def in_collision(self, point, robot_radius=0):
            # Translate point to obstacle frame
            relative = point - self.center
            local = self.inv_rot_matrix @ relative  # Rotate to local frame
    
            half_w = self.width / 2 + robot_radius
            half_h = self.height / 2 + robot_radius
    
            return -half_w <= local[0] <= half_w and -half_h <= local[1] <= half_h

#Plotagem##############################################################################################################################
def plot_path(sol, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    path = sol.get_path()
    path_line, = ax.plot(path[:,0], path[:,1], **kwargs)
    return path_line

def update_path(sol, path_line):
    path = sol.get_path()
    path_line.set_xdata(path[:,0])
    path_line.set_ydata(path[:,1])
    fig = plt.gcf()
    fig.canvas.draw()
    fig.canvas.flush_events()

def plot_environment(environment, uav, ax=None, obstacles_style={}, start_style={}, goal_style={}):
    if ax is None:
        ax = plt.gca()

    ax.set_aspect('equal', adjustable='box')

    # Plot obstacles as circle
    if not 'color' in obstacles_style:
        obstacles_style['color'] = 'k'
    for obstacle in environment.obstacles:
        if isinstance(obstacle, Obstacle):  # Circle
            ax.add_patch(plt.Circle(obstacle.center, obstacle.radius, **obstacles_style))
        elif isinstance(obstacle, RectangularObstacle):  # Rotated rectangle
            lower_left = obstacle.center - np.array([obstacle.width / 2, obstacle.height / 2])
            rect = Rectangle(lower_left, obstacle.width, obstacle.height,
                                angle=np.rad2deg(obstacle.angle),  # rotate in degrees
                                #rotation_point='center',  # rotate around center
                                **obstacles_style)
            ax.add_patch(rect)

    # Plot start
    if not 'color' in start_style:
        start_style['color'] = 'r'
    if not 'markersize' in start_style:
        start_style['markersize'] = 12
    ax.plot(environment.start[0], environment.start[1], 's', **start_style)

    # Plot goal
    if not 'color' in goal_style:
        goal_style['color'] = 'g'
    if not 'markersize' in goal_style:
        goal_style['markersize'] = 12
    ax.plot(environment.goal[0], environment.goal[1], 's', **goal_style)

    # Set axis limits
    ax.set_xlim([0, environment.width])
    ax.set_ylim([0, environment.height])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title("Navigated trajectory of UAV "+str(uav))

#Função custo##########################################################################################################################
START_VIOLATION_PENALTY = 1
GOAL_VIOLATION_PENALTY = 1
ENV_VIOLATION_PENALTY = 0.2
COLLISION_PENALTY = 1

def PathPlanningCost(sol: SplinePath):

    # Get path
    path = sol.get_path()

    # Length of path
    length = sol.environment.path_length(path)

    # Violations of path
    _, details = sol.environment.count_violations(path)

    # Cost
    cost = length

    # Add penalty for start violation
    if details['start_violation']:
        cost *= 1 + START_VIOLATION_PENALTY

    # Add penalty for goal violation
    if details['goal_violation']:
        cost *= 1 + GOAL_VIOLATION_PENALTY

    # Environment violation
    if details['environment_violation']:
        cost *= 1 + details['environment_violation_count']*ENV_VIOLATION_PENALTY

    # Collision violation
    if details['collision_violation']:
        cost *= 1 + details['collision_violation_count']*COLLISION_PENALTY

    # Add details
    details['sol'] = sol
    details['path'] = path
    details['length'] = length
    details['cost'] = cost

    return cost, details

def EnvCostFunction(environment: Environment, num_control_points=10, resolution=100):
    def CostFunction(xy):
        sol = SplinePath.from_list(environment, xy, resolution, normalized=True)
        return PathPlanningCost(sol)

    return CostFunction

# Particle Swarm Optimization############################################################################################################
def PSO(problem, **kwargs):

    max_iter = kwargs.get('max_iter', 100)
    pop_size = kwargs.get('pop_size', 100)
    c1 = kwargs.get('c1', 1.4962)
    c2 = kwargs.get('c2', 1.4962)
    w = kwargs.get('w', 0.7298)
    wdamp = kwargs.get('wdamp', 1.0)
    callback = kwargs.get('callback', None)
    resetting = kwargs.get('resetting', None)

    # Empty Particle Template
    empty_particle = {
        'position': None,
        'velocity': None,
        'cost': None,
        'details': None,
        'best': {
            'position': None,
            'cost': np.inf,
            'details': None,
        },
    }

    # Extract Problem Info
    cost_function = problem['cost_function']
    var_min = problem['var_min']
    var_max = problem['var_max']
    num_var = problem['num_var']

    # Initialize Global Best
    gbest = {
        'position': None,
        'cost': np.inf,
        'details': None,
    }

    # Create Initial Population
    pop = []
    for i in range(0, pop_size):
        pop.append(deepcopy(empty_particle))
        pop[i]['position'] = np.random.uniform(var_min, var_max, num_var)
        pop[i]['velocity'] = np.zeros(num_var)
        pop[i]['cost'], pop[i]['details'] = cost_function(pop[i]['position'])
        pop[i]['best']['position'] = deepcopy(pop[i]['position'])
        pop[i]['best']['cost'] = pop[i]['cost']
        pop[i]['best']['details'] = pop[i]['details']

        if pop[i]['best']['cost'] < gbest['cost']:
            gbest = deepcopy(pop[i]['best'])

    # PSO Loop
    for it in range(0, max_iter):
        do_resetting = resetting and ((it + 1) % resetting == 0)
        if do_resetting:
            print('Resetting particles...')

        for i in range(0, pop_size):

            if do_resetting:
                pop[i]['position'] = np.random.uniform(var_min, var_max, num_var)
                pop[i]['velocity'] = np.zeros(num_var)

            else:
                pop[i]['velocity'] = w*pop[i]['velocity'] \
                    + c1*np.random.rand(num_var)*(pop[i]['best']['position'] - pop[i]['position']) \
                    + c2*np.random.rand(num_var)*(gbest['position'] - pop[i]['position'])

                pop[i]['position'] += pop[i]['velocity']
                pop[i]['position'] = np.clip(pop[i]['position'], var_min, var_max)

            pop[i]['cost'], pop[i]['details'] = cost_function(pop[i]['position'])

            if pop[i]['cost'] < pop[i]['best']['cost']:
                pop[i]['best']['position'] = deepcopy(pop[i]['position'])
                pop[i]['best']['cost'] = pop[i]['cost']
                pop[i]['best']['details'] = pop[i]['details']

                if pop[i]['best']['cost'] < gbest['cost']:
                    gbest = deepcopy(pop[i]['best'])

        w *= wdamp
        print('Iteration {}: Best Cost = {}'.format(it + 1, gbest['cost']))

        if callable(callback):
            callback({
                'it': it + 1,
                'gbest': gbest,
                'pop': pop,
            })

    return gbest, pop

def calcula_trajetoria(id, sender, uav, inicio, destino):
    # Cria ambiente
    par_amb = {
        'width': 1005,
        'height': 859,
        'robot_radius': 30,
        'start': inicio,
        'goal': destino,
    }
    amb = Environment(**par_amb)

    r=3
    # Obstaculos
    r=3
    xi = 0
    yi = 20
    obstaculos = []
    for i in range(0,200,20):
        for j in range(0,200,20):
            obstaculos.append({'center': [xi+i, yi+j], 'radius': r})

    for obs in obstaculos:
        amb.add_obstacle(Obstacle(**obs))
    
    # Add a rotated rectangle
    amb.add_obstacle(RectangularObstacle(center=[7.5, 429.5], width=15, height=859, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[502.5, 851.5], width=1005, height=15, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[997.5, 429.5], width=15, height=859, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[502.5, 7.5], width=1005, height=15, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[10.75, 563.5], width=21.5, height=15, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[221.8, 563.5], width=240, height=15, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[335, 707.5], width=15, height=303, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[542.5, 563.5], width=255, height=15, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[677.5, 707.5], width=15, height=303, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[885, 563.5], width=255, height=15, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[602.25, 204], width=70, height=408, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[709, 377.75], width=275.5, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[820.25, 338], width=70, height=140, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[786.25, 30.25], width=430, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[974.75, 278], width=70, height=556, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[860, 525.75], width=200, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[270, 304], width=167, height=180, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[527.5, 314], width=80, height=180, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[517.5, 45], width=100, height=215, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[30.25, 779], width=70, height=160, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[70, 828.75], width=140, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[304.75, 789], width=70, height=140, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[255, 749.25], width=160, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[215, 532.5], width=240, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[445.5, 528], width=61, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[210, 591], width=230, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[542.5, 591], width=255, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[885, 591], width=255, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[370, 779], width=70, height=160, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[405, 734], width=140, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[600, 734], width=140, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[635, 779], width=70, height=160, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[705, 779], width=70, height=140, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[750, 824], width=160, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[925, 824], width=160, height=70, angle=0))
    amb.add_obstacle(RectangularObstacle(center=[970, 779], width=70, height=140, angle=0))

    # Cria funcao custo
    pontos_controle = 3
    resolucao = 50
    funcao_custo = EnvCostFunction(amb, pontos_controle, resolucao)

    # Problema de otimizacao
    problema = {
        'num_var': 2*pontos_controle,
        'var_min': 0,
        'var_max': 1,
        'cost_function': funcao_custo,
    }

    # Funcao callback
    caminho = None
    imagens=[]
    def callback(data):
        global caminho
        it = data['it']
        sol = data['gbest']['details']['sol']
        if it==1:
            #fig = plt.figure(figsize=[7, 7])
            #plot_environment(amb)
            caminho = plot_path(sol, color='b')
            #plt.grid(True)
            #imagens.append([caminho])
            #plt.show(block=False)
            #fig.savefig('/home/gabryelsr/src/Firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/missoes/caminho_calculado'+str(uav)+'.png')
        elif it == 10:
            update_path(sol, caminho)
            fig = plt.figure(figsize=[7, 7])
            plot_environment(amb, uav)
            caminho = plot_path(sol, color='b')
            #fig.grid()
            #fig.xlabel("X")
            #fig.ylabel("Y")
            #fig.title("Navigated trajectory of UAV "+str(self.uav_num))
            #plt.grid(True)
            #imagens.append([caminho])
            #plt.show(block=False)
            fig.savefig('/home/gabryelsr/src/Firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/missoes/dados/caminho_calculado'+str(uav)+'.png')
        else:
            update_path(sol, caminho)
            #fig = plt.figure(figsize=[7, 7])
            #plot_environment(amb)
            #caminho = plot_path(sol, color='b')
            #plt.grid(True)
            #imagens.append([caminho])
            #plt.show(block=False)

        length = data['gbest']['details']['length']
        #plt.title(f"Iteration: {it}, Length: {length:.2f}")

    # Roda PSO
    parametros_pso = {
        'max_iter': 10,
        'pop_size': 50,
        'c1': 2,
        'c2': 1,
        'w': 0.8,
        'wdamp': 1,
        'resetting': 5,
    }
    print("calculando")
    bestsol, pop = PSO(problema, callback=callback, **parametros_pso)
    caminho = bestsol['details']['path'].tolist()
    x = None
    y = None
    h = 15
    #xl = []
    #yl = []
    pontos = []
    for i in range(len(caminho)):
        x = caminho[i][0]
        y = caminho[i][1]
        #xl.append(x)
        #yl.append(y)
        pontos.append([x,y,h])
    t=datetime.now()
    t=float(t.strftime("%Y%m%d%H%M%S.%f"))
    pontos = [[id], [2], [sender], [uav], [t], pontos] #2 é o código para configurar trajetória
    
    return pontos

class GPS():
	# Iniciando a classe GPS 
	def state_callback(self,data):
		self.cur_state = data
		# Obtem estado de /mavros/state
	def pose_sub_callback(self,pose_sub_data):
		self.current_pose = pose_sub_data
		# Funcao para obter posicaoo atual da FCU
	def gps_callback(self,data):
		self.gps = data
		self.gps_read = True
		# Obtem dados de GPS e seta valor para leitura
	def le_gps(self):
		rospy.init_node('leitura_gps', anonymous=True)	# Inicia no de leitura 
		self.gps_read = False # Na inicializacao a leitura eh setada como falsa
		r = rospy.Rate(10) # Frequencia de comunicacao em 10Hz
		rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.gps_callback) # Dados de GPS
		self.localtarget_received = False 
		r = rospy.Rate(10)
		rospy.Subscriber("/mavros/state", State, self.state_callback)
		rospy.Subscriber("/mavros/global_position/global", NavSatFix, self.gps_callback) 
		while not self.gps_read:
			r.sleep()
		latitude = self.gps.latitude #Informacao de latitude
		longitude = self.gps.longitude #Informacao de longitude
		altitude = self.gps.altitude
		return(latitude,longitude, altitude)

def servidor(rede):
    try:
	    pub = rospy.Publisher('cloud', String, queue_size=10)
    except:
        pass
    rate = rospy.Rate(10) # 10hz
    configuracao=""
    configuracao = rede
    rospy.loginfo(configuracao)
    pub.publish(configuracao)
	# while not rospy.is_shutdown():
	# 	configuracao = rede
	# 	rospy.loginfo(configuracao)
	# 	pub.publish(configuracao)
		#rate.sleep()

class Cloud():
    def __init__(self):
        self.coor_msg = None
        self.coor_msg0 = None
        self.coor_msg1 = None
        self.coor_msg2 = None
        self.coor_msg3 = None
        self.coor_msg4 = None
        self.coor_msg5 = None
        self.coor_msg6 = None
        self.coor_msg7 = None
        self.coor_msg8 = None
        self.coor_msg9 = None

        self.bd = pd.DataFrame(columns=['LAYER', 'NODE', 'REC_TIME', 'RX/TX','ID','TYPE','SENDER','RECEIVER','MSG_TIME', 'CONTENT'])
        self.layer = 'cloud'
    
    def subscreve_coor_cloud(self, coordenadores):
        for coordenador in coordenadores:
            c = coordenador
            if c==0:
                self.cl_sub_c0 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor0_callback)
            elif c==1:
                self.cl_sub_c1 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor1_callback)
            elif c==2:
                self.cl_sub_c2 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor2_callback)
            elif c==3:
                self.cl_sub_c3 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor3_callback)
            elif c==4:
                self.cl_sub_c4 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor4_callback)
            elif c==5:
                self.cl_sub_c5 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor5_callback)
            elif c==6:
                self.cl_sub_c6 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor6_callback)
            elif c==7:
                self.cl_sub_c7 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor7_callback)
            elif c==8:
                self.cl_sub_c8 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor8_callback)
            else:
                self.cl_sub_c9 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor9_callback)
    
    def desinscreve_coor_cloud(self, excoordenadores):
        for excoordenador in excoordenadores:
            c = excoordenador
            if c==0:
                self.cl_sub_c0.unregister()
            elif c==1:
                self.cl_sub_c1.unregister()
            elif c==2:
                self.cl_sub_c2.unregister()
            elif c==3:
                self.cl_sub_c3.unregister()
            elif c==4:
                self.cl_sub_c4.unregister()
            elif c==5:
                self.cl_sub_c5.unregister()
            elif c==6:
                self.cl_sub_c6.unregister()
            elif c==7:
                self.cl_sub_c7.unregister()
            elif c==8:
                self.cl_sub_c8.unregister()
            else:
                self.cl_sub_c9.unregister()
    
    def coor0_callback(self, data):
        self.coor_msg0 = data.data
        self.coor_msg0=json.loads(self.coor_msg0)
    def coor1_callback(self, data):
        self.coor_msg1 = data.data
        self.coor_msg1=json.loads(self.coor_msg1)
    def coor2_callback(self, data):
        self.coor_msg2 = data.data
        self.coor_msg2=json.loads(self.coor_msg2)
    def coor3_callback(self, data):
        self.coor_msg3 = data.data
        self.coor_msg3=json.loads(self.coor_msg3)
    def coor4_callback(self, data):
        self.coor_msg4 = data.data
        self.coor_msg4=json.loads(self.coor_msg4)
    def coor5_callback(self, data):
        self.coor_msg5 = data.data
        self.coor_msg5=json.loads(self.coor_msg5)
    def coor6_callback(self, data):
        self.coor_msg6 = data.data
        self.coor_msg6=json.loads(self.coor_msg6)
    def coor7_callback(self, data):
        self.coor_msg7 = data.data
        self.coor_msg7=json.loads(self.coor_msg7)
    def coor8_callback(self, data):
        self.coor_msg8 = data.data
        self.coor_msg8=json.loads(self.coor_msg8)
    def coor9_callback(self, data):
        self.coor_msg9 = data.data
        self.coor_msg9=json.loads(self.coor_msg9)
    
    def le_coordenador(self, numero):
        t=datetime.now()
        t=float(t.strftime("%Y%m%d%H%M%S.%f"))
        if numero==0:
            self.coor_msg = self.coor_msg0
        elif numero==1:
            self.coor_msg = self.coor_msg1
        elif numero==2:
            self.coor_msg = self.coor_msg2
        elif numero==3:
            self.coor_msg = self.coor_msg3
        elif numero==4:
            self.coor_msg = self.coor_msg4
        elif numero==5:
            self.coor_msg = self.coor_msg5
        elif numero==6:
            self.coor_msg = self.coor_msg6
        elif numero==7:
            self.coor_msg = self.coor_msg7
        elif numero==8:
            self.coor_msg = self.coor_msg8
        else:
            self.coor_msg = self.coor_msg9
        if self.coor_msg is not None:
            try:
                noventrada = {'LAYER': self.layer, 'NODE': 1000, 'REC_TIME': t, 'RX/TX': 'RX',
                            'ID': self.coor_msg[0][0],'TYPE': self.coor_msg[1][0],'SENDER': self.coor_msg[2][0],
                            'RECEIVER': self.coor_msg[3][0],'MSG_TIME': self.coor_msg[4][0], 'CONTENT': self.coor_msg[-1]}
                self.bd.loc[len(self.bd)] = noventrada
            except:
                pass
    
    def cloud_loop(self, cluster1, cluster2, cloudid):

        estrutura = [cluster1, cluster2]
        coor_atual = []
        coor_novo = []
        id = 0
        for cluster in estrutura:
            coor_atual.append(cluster[0])
        self.subscreve_coor_cloud(coor_atual)
        for coor in coor_atual:
            self.le_coordenador(coor)
            print(self.coor_msg)

        t=datetime.now()
        t=float(t.strftime("%Y%m%d%H%M%S.%f"))
        rede=[[id], [0], [cloudid], [9000], [t], [cluster1, cluster2]] #msg id, msg type, sender id, receiver id, time,msg content
        rede_log = str(rede)
        servidor(rede_log)
        t=datetime.now()
        t=float(t.strftime("%Y%m%d%H%M%S.%f"))
        try:
            noventrada = {'LAYER': self.layer, 'NODE': cloudid, 'REC_TIME': t, 'RX/TX': 'TX',
                        'ID': rede[0][0],'TYPE': rede[1][0],'SENDER': rede[2][0],
                        'RECEIVER': rede[3][0],'MSG_TIME': rede[4][0], 'CONTENT': rede[-1]}
            self.bd.loc[len(self.bd)] = noventrada
        except:
            pass
        id = id+1

        i=0
        while i<=100:
            servidor(rede_log)
            i=i+1
            t=datetime.now()
            t=float(t.strftime("%Y%m%d%H%M%S.%f"))
            try:
                noventrada = {'LAYER': self.layer, 'NODE': cloudid, 'REC_TIME': t, 'RX/TX': 'TX',
                                'ID': rede[0][0],'TYPE': rede[1][0],'SENDER': rede[2][0],
                                'RECEIVER': rede[3][0],'MSG_TIME': rede[4][0], 'CONTENT': rede[-1]}
                self.bd.loc[len(self.bd)] = noventrada
            except:
                pass
            time.sleep(0.1)
        trajetorias = []
        for cluster in estrutura:
            for uav in cluster:
                j=0
                pontos = calcula_trajetoria(id, cloudid, uav, [0, 0], [random.randint(20, 200), random.randint(20, 200)])
                trajetorias.append(pontos[-1])
                if cluster == estrutura[-1] and uav == cluster[-1]:
                    with open("/home/gabryelsr/src/Firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/missoes/dados/trajetorias.json", 'w') as f:
                        json.dump(trajetorias, f, indent=2)
                while j<=100:
                    servidor(str(pontos))
                    j=j+1
                    t=datetime.now()
                    t=float(t.strftime("%Y%m%d%H%M%S.%f"))
                    try:
                        noventrada = {'LAYER': self.layer, 'NODE': cloudid, 'REC_TIME': t, 'RX/TX': 'TX',
                                        'ID': rede[0][0],'TYPE': rede[1][0],'SENDER': rede[2][0],
                                        'RECEIVER': rede[3][0],'MSG_TIME': rede[4][0], 'CONTENT': rede[-1]}
                        self.bd.loc[len(self.bd)] = noventrada
                    except:
                        pass
                    time.sleep(0.1)

        num = 0
        maxnum = 300
        while num <= maxnum:

            if num == maxnum:
                random.shuffle(cluster1) 
                random.shuffle(cluster2)

                estrutura = [cluster1, cluster2]
                for cluster in estrutura:
                    coor_novo.append(cluster[0])
                if coor_novo != coor_atual:
                    nenc = list(set(coor_atual) - set(coor_novo))
                    if nenc != []:
                        print(nenc)
                        self.desinscreve_coor_cloud(nenc)
                    novos = list(set(coor_novo) - set(coor_atual))
                    if novos != []:
                        print(novos)
                        self.subscreve_coor_cloud(novos)
                    coor_atual = coor_novo
                    coor_novo = []
                for coor in coor_atual:
                    self.le_coordenador(coor)
                    print(self.coor_msg)

                num2 = 0
                maxnum2 = 100
                while num2 <= maxnum2:
                    id=id+1
                    t=datetime.now()
                    t=float(t.strftime("%Y%m%d%H%M%S.%f"))
                    rede=[[id], [1], [cloudid], [9000], [t], [cluster1, cluster2]]
                    rede_log = str(rede)
                    servidor(rede_log)
                    try:
                        noventrada = {'LAYER': self.layer, 'NODE': cloudid, 'REC_TIME': t, 'RX/TX': 'TX',
                                    'ID': rede[0][0],'TYPE': rede[1][0],'SENDER': rede[2][0],
                                    'RECEIVER': rede[3][0],'MSG_TIME': rede[4][0], 'CONTENT': rede[-1]}
                        self.bd.loc[len(self.bd)] = noventrada
                    except:
                        pass
                    num2 = num2 + 1
                num = 0
                time.sleep(0.1)

            id=id+1
            t=datetime.now()
            t=float(t.strftime("%Y%m%d%H%M%S.%f"))
            rede=[[id], [0], [cloudid], [9000], [t], [cluster1,cluster2]]
            rede_log = str(rede)
            servidor(rede_log)
            try:
                noventrada = {'LAYER': self.layer, 'NODE': cloudid, 'REC_TIME': t, 'RX/TX': 'TX',
                            'ID': rede[0][0],'TYPE': rede[1][0],'SENDER': rede[2][0],
                            'RECEIVER': rede[3][0],'MSG_TIME': rede[4][0], 'CONTENT': rede[-1]}
                self.bd.loc[len(self.bd)] = noventrada
            except:
                pass
            num = num + 1

            for coor in coor_atual:
                self.le_coordenador(coor)
                print(self.coor_msg)
            time.sleep(0.1)

        #salvar dataframe
        self.bd.to_excel('/home/gabryelsr/src/Firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/missoes/dados/banco_de_dados_cloud.xlsx')
        #break

def task_allocation_ARCog_NET(S_Li_t, T_list, W_Li_t, H_Li_t, kappa_Li, omega_Li, L):
    """
    Processo de Alocação de Tarefas no ARCog-NET.
    
    Parâmetros:
        S_Li_t: Estado do UAV no tempo t.
        T_list: Lista de tarefas {T_1, T_2, ..., T_R}.
        W_Li_t: Pesos das decisões (TA, PP, FC).
        H_Li_t: Histórico de desempenho.
        kappa_Li: Memória de pesos.
        omega_Li: Memória de históricos.
        L: Taxa de aprendizado.
    
    Retorna:
        T_alloc: Tarefa alocada.
    """
    T_alloc = None
    U_alloc = -np.inf  # Inicializa com valor mínimo
    N_i = []  # Vizinhos do UAV (ajustar conforme rede)
    
    for T_r in T_list:
        # Inicializa custos para a tarefa T_r
        C_TA = compute_C_TA()  # Função não definida (exemplo)
        if T_r == "T_nav":  # Tarefa de navegação
            C_PP = compute_C_PP()  # Função não definida
            C_FC = compute_C_FC()
        else:
            C_PP = 0
            C_FC = 0
        
        # Atualiza custos baseados em W_Li_t e H_Li_t
        for n, W_Li in enumerate(omega_Li):
            H_Li = kappa_Li[n]
            a = np.argmax(W_Li['w_TA'])  # Índice do maior peso TA
            b = np.argmax(W_Li['w_PP'])
            c = np.argmax(W_Li['w_FC'])
            C_TA_max = H_Li['H_TA'][a]
            C_PP_max = H_Li['H_PP'][b]
            C_FC_max = H_Li['H_FC'][c]
            
            if C_TA <= C_TA_max:
                C_TA = C_TA_max
            if C_PP <= C_PP_max:
                C_PP = C_PP_max
            if C_FC <= C_FC_max:
                C_FC = C_FC_max
        
        # Calcula utilidades
        Phi_mission = compute_Phi_mission(S_Li_t, H_Li_t)
        Phi_network = compute_Phi_network(S_Li_t, H_Li_t)
        Phi_energy = compute_Phi_energy(S_Li_t, H_Li_t)
        Phi_navigation = compute_Phi_navigation(S_Li_t, H_Li_t)
        
        w_TA = W_Li_t['w_TA']
        w_PP = W_Li_t['w_PP']
        w_FC = W_Li_t['w_FC']
        
        U_i_r = (w_TA * (Phi_mission + Phi_network + Phi_energy) + 
                w_PP * w_FC * Phi_navigation)
        
        # Utilidade colaborativa (média dos vizinhos)
        U_collab_r = np.mean([U_i_r for _ in N_i]) if N_i else U_i_r
        U_i_r = U_collab_r
        
        # Atualiza tarefa alocada se utilidade for maior
        if U_i_r > U_alloc:
            T_alloc = T_r
            U_alloc = U_i_r
    
    # Executa a tarefa ou solicita colaboração
    if T_alloc is not None:
        execute_task(T_alloc)
        # Atualiza pesos e histórico (aprendizado)
        delta_C_TA = compute_delta_C()  # Exemplo: variação do custo
        W_Li_t['w_TA'] += L * delta_C_TA
        omega_Li.append(W_Li_t)
        kappa_Li.append(H_Li_t)
    else:
        request_collaboration()  # Função não definida
    
    return T_alloc, W_Li_t, omega_Li, kappa_Li

# --- Funções auxiliares ---
def compute_C_TA(): return np.random.rand()
def compute_Phi_mission(S, H): return S['mission'] * H['performance']
def execute_task(T): print(f"Executando tarefa: {T}")
def request_collaboration(): print("Solicitando colaboração...")

def pso_arcog_net_path_planning(T_nav, rho_Li, V_obj_Li, kappa_Li, omega_Li, t_prev, t_current, 
                               C_min, max_iter, min_error):
    """
    Particle Swarm Optimization for ARCog-NET Navigation Path Planning.
    
    Parameters:
        T_nav: Navigation task (unused in this simplified version but kept for structure)
        rho_Li: Initial trajectory points (list of [x, y] coordinates)
        V_obj_Li: Initial velocities for each point
        kappa_Li: Knowledge base (dictionary with historical data)
        omega_Li: Weight history (list of weight matrices)
        t_prev: Previous time step
        t_current: Current time step
        C_min: Minimum coverage threshold
        max_iter: Maximum iterations
        min_error: Minimum error threshold
    
    Returns:
        Optimized trajectory, updated kappa_Li, updated omega_Li
    """
    error = float('inf')
    iterations = 0
    delta_t = t_current - t_prev
    
    while error > min_error and iterations <= max_iter:
        new_rho_Li = []
        
        # Step 1: Evaluate and potentially replace trajectory points
        for i, P in enumerate(rho_Li):
            C_Li = compute_coverage_change(P, delta_t)  # Placeholder function
            
            if C_Li >= C_min:
                new_rho_Li.append(P)
            else:
                # Find best fitting replacement from knowledge base
                best_fit = None
                best_score = -np.inf
                
                for a, weight_matrix in enumerate(omega_Li):
                    for b, row in enumerate(weight_matrix['W_PP']):
                        for c, w in enumerate(row):
                            current_score = w * kappa_Li[a]['H_PP'][b][c]
                            if current_score > best_score:
                                best_score = current_score
                                best_fit = kappa_Li[a]['H_PP'][b][c]
                
                new_point = best_fit if best_fit is not None else [0, 0]  # Default if no fit
                new_rho_Li.append(new_point)
        
        # Step 2: Update trajectory points using velocity and angle
        for k in range(1, len(new_rho_Li)):
            # Calculate velocity and angle
            dx = new_rho_Li[k][0] - new_rho_Li[k-1][0]
            dy = new_rho_Li[k][1] - new_rho_Li[k-1][1]
            
            V_k = np.sqrt(dx**2 + dy**2) / delta_t
            theta_k = math.atan2(dy, dx)
            
            # Update position
            new_rho_Li[k][0] = (new_rho_Li[k-1][0] + 
                               V_k * math.cos(theta_k) * delta_t)
            new_rho_Li[k][1] = (new_rho_Li[k-1][1] + 
                               V_k * math.sin(theta_k) * delta_t)
            
            # Update error
            current_error = abs(compute_coverage_change(new_rho_Li[k], delta_t) - 
                           compute_coverage_change(new_rho_Li[k-1], delta_t))
            if current_error < error:
                error = current_error
        
        rho_Li = new_rho_Li
        iterations += 1
    
    # Update knowledge base and weight history
    C_PP = compute_coverage_change(rho_Li[-1], delta_t)  # Last point's coverage
    
    new_H_Li = {
        'H_PP': generate_new_H_matrix(C_PP),  # Placeholder
        'timestamp': t_current
    }
    
    new_W_Li = {
        'W_PP': update_weights(omega_Li[-1]['W_PP'], C_PP)  # Placeholder
    }
    
    # Update knowledge base and history
    kappa_Li.append(new_H_Li)
    omega_Li.append(new_W_Li)
    
    return rho_Li, kappa_Li, omega_Li

# Placeholder helper functions
def compute_coverage_change(point, delta_t):
    """Calculate coverage change for a point (simplified)"""
    return np.random.uniform(0.5, 1.5)  # Random value for demonstration

def generate_new_H_matrix(C_PP):
    """Generate new H matrix based on coverage"""
    return np.random.rand(3, 3) * C_PP  # Example 3x3 matrix

def update_weights(old_weights, C_PP):
    """Update weight matrix based on coverage"""
    return old_weights * 0.9 + np.random.rand(*old_weights.shape) * 0.1 * C_PP

def arcog_net_formation_control(e_Li_t, H_Li, W_Li, min_err=0.01, max_it=100):
    """
    ARCog-NET PSO Extension for Formation Control
    
    Parameters:
        e_Li_t: Current formation error (vector)
        H_Li: Historical formation adjustments (list of matrices)
        W_Li: Historical weighting adjustments (list of matrices)
        min_err: Minimum error threshold
        max_it: Maximum iterations
        
    Returns:
        q_best: Best formation adjustment found
        C_FC: Updated formation control cost
        updated_H_Li: Updated history
        updated_W_Li: Updated weights
    """
    # Initialize
    iterations = 0
    q_best = None
    J_p_best = -np.inf
    C_FC = np.zeros_like(e_Li_t)  # Placeholder for FC cost
    
    # PSO parameters
    num_particles = 20
    positions = np.random.randn(num_particles, len(e_Li_t))  # Particle positions
    velocities = np.random.randn(num_particles, len(e_Li_t)) * 0.1
    
    # Main optimization loop
    while np.linalg.norm(e_Li_t) > min_err and iterations < max_it:
        # Update each particle
        for p in range(num_particles):
            # 1. Compute position adjustment (Δq)
            Δq_p = velocities[p]  # Simplified PSO velocity update
            
            # 2. Evaluate fitness (J_p)
            J_p = evaluate_fitness(Δq_p, e_Li_t, H_Li, W_Li)
            
            # 3. Update weights using softmax
            weights = softmax([w['w_FC'] for w in W_Li[-5:]])  # Use last 5 weights
            W_Li.append({'w_FC': weights[-1]})  # Store updated weights
            
            # 4. Update global best if improved
            if J_p > J_p_best:
                J_p_best = J_p
                q_best = Δq_p.copy()
                C_FC = compute_formation_cost(Δq_p, e_Li_t)  # Eq. from paper
        
        iterations += 1
        
        # Update formation error (simplified)
        e_Li_t = update_formation_error(e_Li_t, q_best)
    
    # Update knowledge base
    updated_H_Li = H_Li.copy()
    updated_H_Li.append({
        'Δq': q_best,
        'C_FC': C_FC,
        'iteration': iterations
    })
    
    return q_best, C_FC, updated_H_Li, W_Li

if __name__ == '__main__':

    #velocidade da simulação: export PX4_SIM_SPEED_FACTOR=2 em Firmware
    #configurar latitude: export PX4_HOME_LAT=-26.236411 em Firmware
    #configurar longitude: export PX4_HOME_LON=-42.963770 em Firmware
    #lancar simulacao (h480): make px4_sitl gazebo_typhoon_h480 em Firmware
    #lancar simulacao (iris): make px4_sitl gazebo em Firmware
    #iniciar MavROS com SITL: roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557" em Firmware
    #iniciar MavROS com FCU:
    #1) Em um terminal, iniciar roscore
    #2) QGC>Application Settings>General>Autoconnect to following devices>Desabilitar SiK Radio
    #3) Em outro terminal: roslaunch mavros px4.launch fcu_url:=/dev/ttyUSB0:57600 gcs_url:=udp://@localhost
    #configurar timesync: sudo vim /opt/ros/kinetic/share/mavros/launch/px4_config.yaml
    #inicializar camera: LIBV4LCONTROL_FLAGS=3 cheese
    #topicos: rostopics list

    rospy.init_node('server', anonymous=True)

    nuvem = Cloud()

    #cpp_calculo = rospy.Subscriber("cpp_pontos", String, cpp_callback)
    
    d=5
    h=2.5

    """ Mensagens:
    Posição 0 - Tipo de mensagem
    0 - Configuração de rede
    1 - Reconfiguração de rede
    2 - Configuração de trajetória

    Posição 1 - Emissor/Destinatário
    1000 - Servidor
    0 a 1 - Robôs

    Posição 2 - Mensagem """

    cloudid = 1000 #id do servidor
    cluster1 = [1,2,3,4]
    cluster2 = [5,6,7,8,9]

    while True:
        nuvem.cloud_loop(cluster1, cluster2, cloudid)

    # try:
    #     nuvem.cloud_loop(cluster1, cluster2, cloudid)
    # except rospy.ROSInterruptException:
    #     pass
