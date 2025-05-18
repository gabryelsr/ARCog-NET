import rospy
from mavros_msgs.msg import GlobalPositionTarget, State, PositionTarget, MountControl
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu, NavSatFix, Image
from std_msgs.msg import Float32, Float64, String, Float64MultiArray
import time
from pyquaternion import Quaternion
import math
import threading
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import cv2
from cv_bridge import CvBridge
import os
import json
import random
from datetime import datetime

numero_uav = 2

class Px4Controller:

	def __init__(self):

		self.imu = None
		self.gps = None
		self.local_pose = None
		self.current_state = None
		self.current_heading = None
		self.takeoff_height = 10
		self.local_enu_position = None
		
		self.current_pos_x = None
		self.current_pos_y = None
		self.current_pos_z = None

		self.cur_target_pose = None
		self.global_target = None

		self.camera_msg = None
		self.placa_msg = None
		self.destino_msg = None
		self.velocidade_drone = None
		self.leitura_velocidade_drone=None
		
		self.pos_lider = None

		self.received_new_task = False
		self.arm_state = False
		self.offboard_state = False
		self.received_imu = False
		self.frame = "BODY"

		self.state = None
		
		self.image = None
		self.br = CvBridge()

		self.controle_gimbal = None

		'''
		identidade deste individuo
		'''

		self.uav_num=numero_uav
		self.uav="uav"+str(self.uav_num)
		self.frame_cam="camera"+str(self.uav_num)
		self.rede_configurada = False

		self.cluster_local = [] #esta variavel guarda a estrutura da rede edge local associada ao individuo deste offboard
		self.novo_cluster = [] #esta variavel serve para verificar se houve alteracao na rede edge local
		self.coordenadores = [] #esta variavel guarda a estrutura de coordenadores fog
		self.novos_coordenadores = [] #esta variavel serve para verificar se houve alteracao na rede fog

		'''
		mensagem de cada coordenador ou agente, alem do servidor
		'''
		
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
		self.sou_coor = False
		self.agente_msg = None
		self.agente_msg0 = None
		self.agente_msg1 = None
		self.agente_msg2 = None
		self.agente_msg3 = None
		self.agente_msg4 = None
		self.agente_msg5 = None
		self.agente_msg6 = None
		self.agente_msg7 = None
		self.agente_msg8 = None
		self.agente_msg9 = None
		self.sou_agente = False
		self.cloud_msg = None #mensagem da cloud para a fog (coordenador)
		#self.fog_msg = None #mensagem da fog para a edge (coordenador para agente)
		#self.edge_msg = None #mensagem da edge para a cloud (agente para servidor através do coordenador)

		self.pontos = None
		self.plotou = False

		self.bd = pd.DataFrame(columns=['LAYER', 'NODE', 'REC_TIME', 'RX/TX','ID','TYPE','SENDER','RECEIVER','MSG_TIME', 'CONTENT'])

		'''
		ros subscribers
		'''
		self.local_pose_sub = rospy.Subscriber("/"+self.uav+"/mavros/local_position/pose", PoseStamped, self.local_pose_callback)
		self.mavros_sub = rospy.Subscriber("/"+self.uav+"/mavros/state", State, self.mavros_state_callback)
		self.gps_sub = rospy.Subscriber("/"+self.uav+"/mavros/global_position/global", NavSatFix, self.gps_callback)
		self.imu_sub = rospy.Subscriber("/"+self.uav+"/mavros/imu/data", Imu, self.imu_callback)

		self.set_target_position_sub = rospy.Subscriber("gi/set_pose/position", PoseStamped, self.set_target_position_callback)
		self.set_target_yaw_sub = rospy.Subscriber("gi/set_pose/orientation", Float32, self.set_target_yaw_callback)
		self.custom_activity_sub = rospy.Subscriber("gi/set_activity/type", String, self.custom_activity_callback)

		self.leitura_camera = rospy.Subscriber(self.frame_cam, Image, self.cls_camera_callback)
		self.leitura_placa = rospy.Subscriber('encontra_placa', String, self.cls_placa_callback)
		self.config_destino = rospy.Subscriber("configura_centro", String, self.destino_callback)
		self.velocidade_sub = rospy.Subscriber("/"+self.uav+"/mavros/local_position/velocity", TwistStamped, self.velocidade_callback)

		'''
		ros publishers
		'''
		self.local_target_pub = rospy.Publisher("/"+self.uav+'/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
		self.setpoint_velocidade_pub = rospy.Publisher("/"+self.uav+'/mavros/setpoint_raw/local', PositionTarget, queue_size = 10)
		self.controle_gimbal_pub = rospy.Publisher("/"+self.uav+'/mavros/mount_control/command', MountControl, queue_size = 10)
		self.posicao_uav = rospy.Publisher("/"+self.uav+'/posicao', String, queue_size = 10)

		'''
		ros services
		'''
		self.armService = rospy.ServiceProxy("/"+self.uav+'/mavros/cmd/arming', CommandBool)
		self.flightModeService = rospy.ServiceProxy("/"+self.uav+'/mavros/set_mode', SetMode)
		
		'''
		definicao da estrutura de rede
		'''
		self.rede_estrutura = rospy.Subscriber("cloud", String, self.cloud_callback)

		self.layer = 'indef'

		while self.rede_configurada == False:
			print("Configurando rede")
			if self.cloud_msg is not None:
				if self.cloud_msg[1]==[0] or self.cloud_msg==[1]: #recebeu dados de configuração de rede
					print("Configuração recebida")
					for group in self.cloud_msg[-1]:
						self.coordenadores.append(group[0])
					for i in range(len(self.cloud_msg[-1])):
						# if i==0: #na primeira posição do array de configuração de rede esta o tipo de ação (configuração ou reconfiguração)
						# 	print("Reconfigurando rede")
						# 	pass
						# else:
						# 	print("Configurando rede")
						if (self.uav_num in self.cloud_msg[-1][i]):
							self.cluster_local = self.cloud_msg[-1][i]
							print(self.cluster_local)
							for j in range(len(self.cluster_local)):
								#coor="c_uav_"
								#agen="a_uav_"
								if self.uav_num==self.cluster_local[0]: #verifica se o uav é um coordenador
									print("Sou um coordenador")
									self.sou_coor = True
									self.sou_agente = False
									self.layer = 'fog'
									#iniciar o publisher deste uav se ele for um coordenador
									self.registra_coordenador()
									for a in self.cluster_local[1:]:
										print("Subscrevendo a agente" + str(a))
										#subscreve aos topicos dos agentes edge ligados a ele
										self.subscreve_coordenador_agente(a)
									for c in self.coordenadores:
										if c==self.uav_num:
											pass
										else:
											print("Subscrevendo a coordenador" + str(c))
											#subscreve a outros coordenadores fog
											self.subscreve_coor_coordenador(c)
									print("Configuracao concluida")
									self.rede_configurada = True
								else: #o uav é um agente
									print("Sou um agente")
									self.sou_coor = False
									self.sou_agente = True
									self.layer = 'edge'
									self.rede_estrutura.unregister() #para de se comunicar com o servidor
									self.registra_agente()
									c = self.cluster_local[0] #descobre qual o coordenador deste agente
									#subscreve aos topicos dos coordenadores designados a ele
									self.subscreve_agente_coordenador(c)
									print("Configuração concluida")
									self.rede_configurada = True

		print("Robô inicializado!")
	
	def reconfigura_rede(self, id, tipo, rec):
		if self.cloud_msg is not None:
			if self.cloud_msg[1]==[1]: #se for comando de reconfiguração da rede
				print("Reconfigurando rede")
				for cluster in self.cloud_msg[-1]:
						if self.uav_num in cluster: #na primeira posição do array de configuração de rede esta o tipo de ação (configuração ou reconfiguração)
							self.novo_cluster = cluster
							if self.novo_cluster != self.cluster_local: #verifica se houve mudança no cluster
								if self.uav_num == self.novo_cluster[0] and self.sou_coor == True and self.sou_agente == False: #o robo ainda é um coordenador
									dif = list(set(self.cluster_local) - set(self.novo_cluster))
									if dif != []:
										for nenc in dif:
											self.desinscreve_coordenador_agente(nenc)
									nov = list(set(self.novo_cluster) - set(self.cluster_local))
									if nov != []:
										for nenc in nov:
											self.subscreve_coordenador_agente(nenc)
									self.cluster_local = self.novo_cluster
									self.sou_coor = True
									self.sou_agente = False
									self.layer = 'fog'
									self.novos_coordenadores = []
									for group in self.cloud_msg[-1]:
											self.novos_coordenadores.append(group[0])
									if self.novos_coordenadores == self.coordenadores:
										pass
									else:
										if self.sou_coor == True and self.sou_agente == False:
											difc = list(set(self.coordenadores) - set(self.novos_coordenadores))
											if difc != []:
												for cnenc in difc:
													if cnenc == self.uav_num:
														pass
													else:
														self.desinscreve_coor_coordenador(cnenc)
											novc = list(set(self.novos_coordenadores) - set(self.coordenadores))
											if novc != []:
												for cnenc in novc:
													if cnenc == self.uav_num:
														pass
													else:
														self.subscreve_coor_coordenador(cnenc)
										self.coordenadores = self.novos_coordenadores
								elif self.uav_num == self.novo_cluster[0] and self.sou_coor == False and self.sou_agente == True: #o robo era agente e vai virar coordenador
									self.rede_estrutura = rospy.Subscriber("cloud", String, self.cloud_callback)
									self.registra_coordenador()
									ex_coor = self.cluster_local[0]
									self.desinscreve_agente_coordenador(ex_coor)
									for a in self.novo_cluster[1:]:
										self.subscreve_coordenador_agente(a)
									self.cluster_local = self.novo_cluster
									self.sou_coor = True
									self.sou_agente = False
									self.layer = 'fog'
									self.novos_coordenadores = []
									for group in self.cloud_msg[-1]:
											self.novos_coordenadores.append(group[0])
									if self.novos_coordenadores == self.coordenadores:
										pass
									else:
										if self.sou_coor == True and self.sou_agente == False:
											novc = list(set(self.novos_coordenadores) - set(self.coordenadores))
											if novc != []:
												for cnenc in novc:
													if cnenc == self.uav_num:
														pass
													else:
														self.subscreve_coor_coordenador(cnenc)
										self.coordenadores = self.novos_coordenadores
								elif self.uav_num != self.novo_cluster[0] and self.sou_coor == True and self.sou_agente == False: #o robo era coordenador e vai virar agente
									num = 0
									while num <= 300: #descarrega dados aos agentes para garantir recebimento da mensagem de reconfiguração
										self.publica_coor(id, tipo, rec)
										num = num + 1
									for a in self.cluster_local[1:]:
										self.desinscreve_coordenador_agente(a)
									self.registra_agente()
									novo_coor = self.novo_cluster[0]
									self.subscreve_agente_coordenador(novo_coor)
									self.rede_estrutura.unregister()
									self.cluster_local = self.novo_cluster
									self.sou_coor = False
									self.sou_agente = True
									self.layer = 'edge'
									self.novos_coordenadores = []
									for group in self.cloud_msg[-1]:
											self.novos_coordenadores.append(group[0])
									if self.novos_coordenadores == self.coordenadores:
										pass
									else:
										if self.sou_coor == True and self.sou_agente == False:
											difc = list(set(self.coordenadores) - set(self.novos_coordenadores))
											if difc != []:
												for cnenc in difc:
													if cnenc == self.uav_num:
														pass
													else:
														self.desinscreve_coor_coordenador(cnenc)
										self.coordenadores = self.novos_coordenadores
								else: #o robo ainda é um agente
									if self.novo_cluster[0] != self.cluster_local[0]:
										self.desinscreve_agente_coordenador(self.cluster_local[0])
										self.subscreve_agente_coordenador(self.novo_cluster[0])
									self.cluster_local = self.novo_cluster
									self.sou_coor = False
									self.sou_agente = True
									self.layer = 'edge'
									self.coordenadores = self.novos_coordenadores
							
	def registra_coordenador(self):
		if self.uav_num==0:
			self.c_uav0_pub = rospy.Publisher("/"+self.uav+"/coor_msg", String, queue_size = 10)
		elif self.uav_num==1:
			self.c_uav1_pub = rospy.Publisher("/"+self.uav+"/coor_msg", String, queue_size = 10)
		elif self.uav_num==2:
			self.c_uav2_pub = rospy.Publisher("/"+self.uav+"/coor_msg", String, queue_size = 10)
		elif self.uav_num==3:
			self.c_uav3_pub = rospy.Publisher("/"+self.uav+"/coor_msg", String, queue_size = 10)
		elif self.uav_num==4:
			self.c_uav4_pub = rospy.Publisher("/"+self.uav+"/coor_msg", String, queue_size = 10)
		elif self.uav_num==5:
			self.c_uav5_pub = rospy.Publisher("/"+self.uav+"/coor_msg", String, queue_size = 10)
		elif self.uav_num==6:
			self.c_uav6_pub = rospy.Publisher("/"+self.uav+"/coor_msg", String, queue_size = 10)
		elif self.uav_num==7:
			self.c_uav7_pub = rospy.Publisher("/"+self.uav+"/coor_msg", String, queue_size = 10)
		elif self.uav_num==8:
			self.c_uav8_pub = rospy.Publisher("/"+self.uav+"/coor_msg", String, queue_size = 10)
		else:
			self.c_uav9_pub = rospy.Publisher("/"+self.uav+"/coor_msg", String, queue_size = 10)
	
	def subscreve_coordenador_agente(self, agente):
		a = agente
		if a==0:
			self.c_sub_a0 = rospy.Subscriber("/uav"+str(a)+"/agente_msg", String, self.agente0_callback)
		elif a==1:
			self.c_sub_a1 = rospy.Subscriber("/uav"+str(a)+"/agente_msg", String, self.agente1_callback)
		elif a==2:
			self.c_sub_a2 = rospy.Subscriber("/uav"+str(a)+"/agente_msg", String, self.agente2_callback)
		elif a==3:
			self.c_sub_a3 = rospy.Subscriber("/uav"+str(a)+"/agente_msg", String, self.agente3_callback)
		elif a==4:
			self.c_sub_a4 = rospy.Subscriber("/uav"+str(a)+"/agente_msg", String, self.agente4_callback)
		elif a==5:
			self.c_sub_a5 = rospy.Subscriber("/uav"+str(a)+"/agente_msg", String, self.agente5_callback)
		elif a==6:
			self.c_sub_a6 = rospy.Subscriber("/uav"+str(a)+"/agente_msg", String, self.agente6_callback)
		elif a==7:
			self.c_sub_a7 = rospy.Subscriber("/uav"+str(a)+"/agente_msg", String, self.agente7_callback)
		elif a==8:
			self.c_sub_a8 = rospy.Subscriber("/uav"+str(a)+"/agente_msg", String, self.agente8_callback)
		else:
			self.c_sub_a9 = rospy.Subscriber("/uav"+str(a)+"/agente_msg", String, self.agente9_callback)

	def desinscreve_coordenador_agente(self, agente):
		a = agente
		try:
			if a==0:
				self.c_sub_a0.unregister()
			elif a==1:
				self.c_sub_a1.unregister()
			elif a==2:
				self.c_sub_a2.unregister()
			elif a==3:
				self.c_sub_a3.unregister()
			elif a==4:
				self.c_sub_a4.unregister()
			elif a==5:
				self.c_sub_a5.unregister()
			elif a==6:
				self.c_sub_a6.unregister()
			elif a==7:
				self.c_sub_a7.unregister()
			elif a==8:
				self.c_sub_a8.unregister()
			else:
				self.c_sub_a9.unregister()
		except:
			pass

	def subscreve_coor_coordenador(self, oth_coor):
		c = oth_coor
		if c==0:
			self.c_sub_c0 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor0_callback)
		elif c==1:
			self.c_sub_c1 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor1_callback)
		elif c==2:
			self.c_sub_c2 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor2_callback)
		elif c==3:
			self.c_sub_c3 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor3_callback)
		elif c==4:
			self.c_sub_c4 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor4_callback)
		elif c==5:
			self.c_sub_c5 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor5_callback)
		elif c==6:
			self.c_sub_c6 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor6_callback)
		elif c==7:
			self.c_sub_c7 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor7_callback)
		elif c==8:
			self.c_sub_c8 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor8_callback)
		else:
			self.c_sub_c9 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor9_callback)

	def desinscreve_coor_coordenador(self, oth_coor):
		c = oth_coor
		try:
			if c==0:
				self.c_sub_c0.unregister()
			elif c==1:
				self.c_sub_c1.unregister()
			elif c==2:
				self.c_sub_c2.unregister()
			elif c==3:
				self.c_sub_c3.unregister()
			elif c==4:
				self.c_sub_c4.unregister()
			elif c==5:
				self.c_sub_c5.unregister()
			elif c==6:
				self.c_sub_c6.unregister()
			elif c==7:
				self.c_sub_c7.unregister()
			elif c==8:
				self.c_sub_c8.unregister()
			else:
				self.c_sub_c9.unregister()
		except:
			pass
		
	def registra_agente(self):
		if self.uav_num==0:
			self.a_uav0_pub = rospy.Publisher("/"+self.uav+"/agente_msg", String, queue_size = 10)
		elif self.uav_num==1:
			self.a_uav1_pub = rospy.Publisher("/"+self.uav+"/agente_msg", String, queue_size = 10)
		elif self.uav_num==2:
			self.a_uav2_pub = rospy.Publisher("/"+self.uav+"/agente_msg", String, queue_size = 10)
		elif self.uav_num==3:
			self.a_uav3_pub = rospy.Publisher("/"+self.uav+"/agente_msg", String, queue_size = 10)
		elif self.uav_num==4:
			self.a_uav4_pub = rospy.Publisher("/"+self.uav+"/agente_msg", String, queue_size = 10)
		elif self.uav_num==5:
			self.a_uav5_pub = rospy.Publisher("/"+self.uav+"/agente_msg", String, queue_size = 10)
		elif self.uav_num==6:
			self.a_uav6_pub = rospy.Publisher("/"+self.uav+"/agente_msg", String, queue_size = 10)
		elif self.uav_num==7:
			self.a_uav7_pub = rospy.Publisher("/"+self.uav+"/agente_msg", String, queue_size = 10)
		elif self.uav_num==8:
			self.a_uav8_pub = rospy.Publisher("/"+self.uav+"/agente_msg", String, queue_size = 10)
		else:
			self.a_uav9_pub = rospy.Publisher("/"+self.uav+"/agente_msg", String, queue_size = 10)
	
	def subscreve_agente_coordenador(self, coordenador):
		c = coordenador
		if c==0:
			self.a_sub_c0 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor0_callback)
		elif c==1:
			self.a_sub_c1 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor1_callback)
		elif c==2:
			self.a_sub_c2 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor2_callback)
		elif c==3:
			self.a_sub_c3 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor3_callback)
		elif c==4:
			self.a_sub_c4 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor4_callback)
		elif c==5:
			self.a_sub_c5 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor5_callback)
		elif c==6:
			self.a_sub_c6 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor6_callback)
		elif c==7:
			self.a_sub_c7 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor7_callback)
		elif c==8:
			self.a_sub_c8 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor8_callback)
		else:
			self.a_sub_c9 = rospy.Subscriber("/uav"+str(c)+"/coor_msg", String, self.coor9_callback)
	
	def desinscreve_agente_coordenador(self, coordenador):
		c = coordenador
		try:
			if c==0:
				self.a_sub_c0.unregister()
			elif c==1:
				self.a_sub_c1.unregister()
			elif c==2:
				self.a_sub_c2.unregister()
			elif c==3:
				self.a_sub_c3.unregister()
			elif c==4:
				self.a_sub_c4.unregister()
			elif c==5:
				self.a_sub_c5.unregister()
			elif c==6:
				self.a_sub_c6.unregister()
			elif c==7:
				self.a_sub_c7.unregister()
			elif c==8:
				self.a_sub_c8.unregister()
			else:
				self.a_sub_c9.unregister()
		except:
			pass
	
	def start(self):
		#rospy.init_node("offboard_node")
		for i in range(10):
			if self.current_heading is not None:
				break
			else:
				print("Waiting for initialization.")
				time.sleep(0.5)
		self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)

		#print ("self.cur_target_pose:", self.cur_target_pose, type(self.cur_target_pose))

		for i in range(10):
			self.local_target_pub.publish(self.cur_target_pose)
			self.arm_state = self.arm()
			self.offboard_state = self.offboard()
			time.sleep(0.2)


		if self.takeoff_detection():
			print("Vehicle Took Off!")

		else:
			print("Vehicle Took Off Failed!")
			return

		'''
		main ROS thread
		'''
		while self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):

			self.local_target_pub.publish(self.cur_target_pose)

			if (self.state == "LAND") and (self.local_pose.pose.position.z < 0.15):

				if(self.disarm()):

					self.state = "DISARMED"


			time.sleep(0.1)

# std_msgs/Header header

# uint8 coordinate_frame
# uint8 FRAME_LOCAL_NED = 1
# uint8 FRAME_LOCAL_OFFSET_NED = 7
# uint8 FRAME_BODY_NED = 8
# uint8 FRAME_BODY_OFFSET_NED = 9

# uint16 type_mask
# uint16 IGNORE_PX = 1 # Position ignore flags
# uint16 IGNORE_PY = 2
# uint16 IGNORE_PZ = 4
# uint16 IGNORE_VX = 8 # Velocity vector ignore flags
# uint16 IGNORE_VY = 16
# uint16 IGNORE_VZ = 32
# uint16 IGNORE_AFX = 64 # Acceleration/Force vector ignore flags
# uint16 IGNORE_AFY = 128
# uint16 IGNORE_AFZ = 256
# uint16 FORCE = 512 # Force in af vector flag
# uint16 IGNORE_YAW = 1024
# uint16 IGNORE_YAW_RATE = 2048

# geometry_msgs/Point position
# geometry_msgs/Vector3 velocity
# geometry_msgs/Vector3 acceleration_or_force
# float32 yaw
# float32 yaw_rate


	def construct_target(self, x, y, z, yaw, yaw_rate = 0.00):
		target_raw_pose = PositionTarget()
		target_raw_pose.header.stamp = rospy.Time.now()

		target_raw_pose.coordinate_frame = 1

		difx=0
		dify=0
		""" if self.uav_num==1:
			difx=0
			dify=1
		elif self.uav_num==2:
			difx=0
			dify=2
		elif self.uav_num==3:
			difx=0
			dify=3
		elif self.uav_num==4:
			difx=1
			dify=1
		elif self.uav_num==5:
			difx=1
			dify=2
		elif self.uav_num==6:
			difx=1
			dify=3
		elif self.uav_num==7:
			difx=2
			dify=1
		elif self.uav_num==8:
			difx=2
			dify=2
		elif self.uav_num==9:
			difx=2
			dify=3 """

		target_raw_pose.position.x = x + difx
		target_raw_pose.position.y = y + dify
		target_raw_pose.position.z = z

		# target_raw_pose.velocity.x = 1
		# target_raw_pose.velocity.y = 1
		# target_raw_pose.velocity.z = 1

		# target_raw_pose.acceleration_or_force.x = 0.01
		# target_raw_pose.acceleration_or_force.y = 0.01
		# target_raw_pose.acceleration_or_force.z = 0.01

		target_raw_pose.type_mask = PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ \
									+ PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ \
									+ PositionTarget.FORCE

		target_raw_pose.yaw = yaw
		target_raw_pose.yaw_rate = yaw_rate

		return target_raw_pose

	def controle_orientacao(self, yaw, h, yaw_rate = 0.00):
		print('ENTROU NO CONTROLE DE ORIENTACAO')
		target_raw_pose = PositionTarget()
		target_raw_pose.header.stamp = rospy.Time.now()

		target_raw_pose.coordinate_frame = 1

		target_raw_pose.position.x = self.local_pose.pose.position.x
		target_raw_pose.position.y = self.local_pose.pose.position.y
		target_raw_pose.position.z = h


		target_raw_pose.type_mask = PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ \
									+ PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ \
									+ PositionTarget.FORCE

		target_raw_pose.yaw = np.radians(yaw)
		target_raw_pose.yaw_rate = yaw_rate

		return target_raw_pose

	def calcula_velocidade_drone(self, v, ang):
		#ang=self.current_heading*180/np.pi
		print('ENTROU NO CALCULO DE VELOCIDADE')
		if ang==0:
			orientacao='L'
			vx=v*np.cos(np.radians(ang))
			vy=v*np.sin(np.radians(ang))
		elif ang>0 and ang<90:
			orientacao='NE'
			vx=v*np.cos(np.radians(ang))
			vy=v*np.sin(np.radians(ang))
		elif ang==90:
			orientacao='N'
			vx=v*np.cos(np.radians(ang))
			vy=v*np.sin(np.radians(ang))
		elif ang>90 and ang<180:
			orientacao='NO'
			ang=180-ang
			vx=-1*v*np.cos(np.radians(ang))
			vy=v*np.sin(np.radians(ang))
		elif ang==180 or ang==-180:
			orientacao='O'
			vx=-1*v*np.cos(np.radians(ang))
			vy=v*np.sin(np.radians(ang))
		elif ang<0 and ang>-90:
			orientacao='SE'
			ang=-1*ang
			vx=v*np.cos(np.radians(ang))
			vy=-1*v*np.sin(np.radians(ang))
		elif ang==-90:
			orientacao='S'
			ang=-1*ang
			vx=-1*v*np.cos(np.radians(ang))
			vy=-1*v*np.sin(np.radians(ang))
		else:
			orientacao='SO'
			ang=-1*(-180-ang)
			vx=-1*v*np.cos(np.radians(ang))
			vy=-1*v*np.sin(np.radians(ang))
		return orientacao, vx, vy

	def controle_velocidade(self, vx, vy, yaw_rate=0.00):
			print('ENTROU NO CONTROLE DE VELOCIDADE')
			drone_vel = PositionTarget()

			#yaw=yaw*np.pi/180

			#vx = v*np.cos(yaw)
			#vy = v*np.sin(yaw)
			#drone_vel.velocity.x = vx
			#drone_vel.velocity.y = vy

			drone_vel.header.stamp = rospy.Time.now()

			drone_vel.coordinate_frame = 1

			#z=30
			#drone_vel.position.z = z

			drone_vel.velocity.x = vx
			drone_vel.velocity.y = vy
			# target_raw_pose.velocity.z = vz

			# target_raw_pose.acceleration_or_force.x = afx
			# target_raw_pose.acceleration_or_force.y = afy
			# target_raw_pose.acceleration_or_force.z = afz

			drone_vel.type_mask = PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ \
										+ PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ \
										+ PositionTarget.IGNORE_YAW + PositionTarget.IGNORE_YAW_RATE + PositionTarget.FORCE

			#drone_vel.yaw = self.current_heading
			#drone_vel.yaw_rate = yaw_rate
			#print(vx)
			#print(vy)

			return drone_vel

	def controle_gimbal_camera(self, r, p, y):
			gb_controle = MountControl()
		
			gb_controle.header.stamp = rospy.Time.now()
			gb_controle.header.frame_id="map"
			gb_controle.mode=2
		
			gb_controle.roll=r
			gb_controle.pitch=p
			gb_controle.yaw=y
		
			return gb_controle

	'''
	cur_p : poseStamped
	target_p: positionTarget
	'''
	def position_distance(self, cur_p, target_p, threshold=0.1):
		delta_x = math.fabs(cur_p.pose.position.x - target_p.position.x)
		delta_y = math.fabs(cur_p.pose.position.y - target_p.position.y)
		delta_z = math.fabs(cur_p.pose.position.z - target_p.position.z)

		if (delta_x + delta_y + delta_z < threshold):
			return True
		else:
			return False
			
	def lider_posicao_callback(self, data):
		self.pos_lider = data.data

	def cls_camera_callback(self, data):
		#self.camera_msg = data.data
		self.image = self.br.imgmsg_to_cv2(data)
			
	def cls_placa_callback(self, data):
		self.placa_msg = data.data
		#camera_msg=data.data
		if self.placa_msg=='Placa Encontrada!':
			self.placa_msg='Parando Etapa 2!'
		else:
			self.placa_msg=' '

	def destino_callback(self, data):
		self.destino_msg=data.data
		self.destino_msg = [float(coordenadas_centro) for coordenadas_centro in self.destino_msg.split(",")]

	def local_pose_callback(self, msg):
		self.local_pose = msg
		self.local_enu_position = msg

	def velocidade_callback(self, msg):
		self.leitura_velocidade_drone = msg

	def mavros_state_callback(self, msg):
		self.mavros_state = msg.mode

	def imu_callback(self, msg):
		global global_imu, current_heading
		self.imu = msg

		self.current_heading = self.q2yaw(self.imu.orientation)

		self.received_imu = True

	def gps_callback(self, msg):
		self.gps = msg
		
	'''
	callbacks de rede
	'''
	
	def cloud_callback(self, data):
		self.cloud_msg = data.data
		self.cloud_msg=json.loads(self.cloud_msg)
		#print(self.cloud_msg)
		t=datetime.now()
		t=float(t.strftime("%Y%m%d%H%M%S.%f"))
		if self.cloud_msg is not None:
			try:
				noventrada = {'LAYER': self.layer, 'NODE': self.uav_num, 'REC_TIME': t, 'RX/TX': 'RX',
								'ID': self.cloud_msg[0][0],'TYPE': self.cloud_msg[1][0],'SENDER': self.cloud_msg[2][0],
								'RECEIVER': self.cloud_msg[3][0],'MSG_TIME': self.cloud_msg[4][0], 'CONTENT': self.cloud_msg[-1]}
				self.bd.loc[len(self.bd)] = noventrada
			except:
				pass

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

	def agente0_callback(self, data):
		self.agente_msg0 = data.data
		self.agente_msg0=json.loads(self.agente_msg0)
	def agente1_callback(self, data):
		self.agente_msg1 = data.data
		self.agente_msg1=json.loads(self.agente_msg1)
	def agente2_callback(self, data):
		self.agente_msg2 = data.data
		self.agente_msg2=json.loads(self.agente_msg2)
	def agente3_callback(self, data):
		self.agente_msg3 = data.data
		self.agente_msg3=json.loads(self.agente_msg3)
	def agente4_callback(self, data):
		self.agente_msg4 = data.data
		self.agente_msg4=json.loads(self.agente_msg4)
	def agente5_callback(self, data):
		self.agente_msg5 = data.data
		self.agente_msg5=json.loads(self.agente_msg5)
	def agente6_callback(self, data):
		self.agente_msg6 = data.data
		self.agente_msg6=json.loads(self.agente_msg6)
	def agente7_callback(self, data):
		self.agente_msg7 = data.data
		self.agente_msg7=json.loads(self.agente_msg7)
	def agente8_callback(self, data):
		self.agente_msg8 = data.data
		self.agente_msg8=json.loads(self.agente_msg8)
	def agente9_callback(self, data):
		self.agente_msg9 = data.data
		self.agente_msg9=json.loads(self.agente_msg9)

	def publica_coor(self, id, tipo, rec):
		#rate = rospy.Rate(10)
		#coor="c_uav_"
		if not rospy.is_shutdown() and self.local_pose is not None:
			self.current_pos_x = self.local_pose.pose.position.x
			self.current_pos_y = self.local_pose.pose.position.y
			self.current_pos_z = self.local_pose.pose.position.z
			t=datetime.now()
			t=float(t.strftime("%Y%m%d%H%M%S.%f"))
			log = [[id], [tipo], [self.uav_num],[rec], [t], 
					[[self.current_pos_x,
					self.current_pos_y,
					self.current_pos_z,
					self.current_heading],
					self.cloud_msg]]
			content = log[-1]
			log = str(log)
			#rospy.loginfo(log)
			#globals()[coor + str(self.uav_num)+"_pub"].publish(log)
			if self.uav_num == 0:
				self.c_uav0_pub.publish(log)
			elif self.uav_num == 1:
				self.c_uav1_pub.publish(log)
			elif self.uav_num == 2:
				self.c_uav2_pub.publish(log)
			elif self.uav_num == 3:
				self.c_uav3_pub.publish(log)
			elif self.uav_num == 4:
				self.c_uav4_pub.publish(log)
			elif self.uav_num == 5:
				self.c_uav5_pub.publish(log)
			elif self.uav_num == 6:
				self.c_uav6_pub.publish(log)
			elif self.uav_num == 7:
				self.c_uav7_pub.publish(log)
			elif self.uav_num == 8:
				self.c_uav8_pub.publish(log)
			else:
				self.c_uav9_pub.publish(log)
			print(log)
			if self.cloud_msg is not None:
				try:
					noventrada = {'LAYER': self.layer, 'NODE': self.uav_num, 'REC_TIME': t, 'RX/TX': 'TX',
								'ID': id,'TYPE': tipo,'SENDER': self.uav_num,'RECEIVER': rec,'MSG_TIME': t, 
								'CONTENT': content}
					self.bd.loc[len(self.bd)] = noventrada
				except:
					pass
			# if self.cloud_msg is not None:
			# 	print(self.cloud_msg[1])
			# else:
			# 	print(self.cloud_msg)
			#rate.sleep()						

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
		#O bloco abaixo serve para a edge receber mensagem da cloud através da fog:
		if self.sou_coor == True:
			try:
				noventrada = {'LAYER': self.layer, 'NODE': self.uav_num, 'REC_TIME': t, 'RX/TX': 'RX',
							'ID': self.coor_msg[0][0],'TYPE': self.coor_msg[1][0],'SENDER': self.coor_msg[2][0],
							'RECEIVER': self.coor_msg[3][0],'MSG_TIME': self.coor_msg[4][0], 'CONTENT': self.coor_msg[-1]}
				self.bd.loc[len(self.bd)] = noventrada
			except:
				pass
		else:
			if self.coor_msg is not None:
				try:
					noventrada = {'LAYER': self.layer, 'NODE': self.uav_num, 'REC_TIME': t, 'RX/TX': 'RX',
							'ID': self.coor_msg[0][0],'TYPE': self.coor_msg[1][0],'SENDER': self.coor_msg[2][0],
							'RECEIVER': self.coor_msg[3][0],'MSG_TIME': self.coor_msg[4][0], 'CONTENT': self.coor_msg[-1]}
					self.bd.loc[len(self.bd)] = noventrada
					self.cloud_msg = self.coor_msg[-1][-1]
					noventrada = {'LAYER': self.layer, 'NODE': self.uav_num, 'REC_TIME': t, 'RX/TX': 'RX',
									'ID': self.coor_msg[0][0],'TYPE': self.coor_msg[1][0],'SENDER': self.coor_msg[2][0],
									'RECEIVER': self.coor_msg[3][0],'MSG_TIME': self.coor_msg[4][0], 'CONTENT': self.coor_msg[-1]}
					self.bd.loc[len(self.bd)] = noventrada
				except:
					pass
	
	def publica_agente(self, id, tipo, rec):
		#rate = rospy.Rate(10)
		#agen="a_uav_"
		if not rospy.is_shutdown() and self.local_pose is not None:
			self.current_pos_x = self.local_pose.pose.position.x
			self.current_pos_y = self.local_pose.pose.position.y
			self.current_pos_z = self.local_pose.pose.position.z
			t=datetime.now()
			t=float(t.strftime("%Y%m%d%H%M%S.%f"))
			log = [[id], [tipo], [self.uav_num],[rec], [t],
				   [self.current_pos_x,
			       self.current_pos_y,
			       self.current_pos_z,
			       self.current_heading]]
			content = log[-1]
			log = str(log)
			#rospy.loginfo(log)
			#globals()[agen + str(self.uav_num)+"_pub"].publish(log)
			if self.uav_num==0:
				self.a_uav0_pub.publish(log)
			elif self.uav_num==1:
				self.a_uav1_pub.publish(log)
			elif self.uav_num == 2:
				self.a_uav2_pub.publish(log)
			elif self.uav_num == 3:
				self.a_uav3_pub.publish(log)
			elif self.uav_num == 4:
				self.a_uav4_pub.publish(log)
			elif self.uav_num == 5:
				self.a_uav5_pub.publish(log)
			elif self.uav_num == 6:
				self.a_uav6_pub.publish(log)
			elif self.uav_num == 7:
				self.a_uav7_pub.publish(log)
			elif self.uav_num == 8:
				self.a_uav8_pub.publish(log)
			else:
				self.a_uav9_pub.publish(log)	
			# if self.cloud_msg is not None:
			# 	print(self.cloud_msg[1])
			# else:
			# 	print(self.cloud_msg)
			#rate.sleep()
			if self.cloud_msg is not None:
				try:
					noventrada = {'LAYER': self.layer, 'NODE': self.uav_num, 'REC_TIME': t, 'RX/TX': 'TX',
								'ID': id,'TYPE': tipo,'SENDER': self.uav_num,'RECEIVER': rec,'MSG_TIME': t, 
								'CONTENT': content}
					self.bd.loc[len(self.bd)] = noventrada
				except:
					pass
	
	def le_agente(self, numero):
		t=datetime.now()
		t=float(t.strftime("%Y%m%d%H%M%S.%f"))
		if numero==0:
			self.agente_msg = self.agente_msg0
		elif numero==1:
			self.agente_msg = self.agente_msg1
		elif numero==2:
			self.agente_msg = self.agente_msg2
		elif numero==3:
			self.agente_msg = self.agente_msg3
		elif numero==4:
			self.agente_msg = self.agente_msg4
		elif numero==5:
			self.agente_msg = self.agente_msg5
		elif numero==6:
			self.agente_msg = self.agente_msg6
		elif numero==7:
			self.agente_msg = self.agente_msg7
		elif numero==8:
			self.agente_msg = self.agente_msg8
		else:
			self.agente_msg = self.agente_msg9

		if self.agente_msg is not None:
			try:
				noventrada = {'LAYER': self.layer, 'NODE': self.uav_num, 'REC_TIME': t, 'RX/TX': "RX",
								'ID': self.agente_msg[0][0],'TYPE': self.agente_msg[1][0],'SENDER': self.agente_msg[2][0],
								'RECEIVER': self.agente_msg[3][0],'MSG_TIME': self.agente_msg[4][0], 'CONTENT': self.agente_msg[-1]}
				self.bd.loc[len(self.bd)] = noventrada
			except:
				pass

	def sitl(self):
		while not rospy.is_shutdown():
			if self.cluster_local == []:
				print("Rede ainda não definida --iniciando configuração")
			else:
				print("Rede definida")
				#print(globals().keys())
				if self.uav_num == self.cluster_local[0]:
					print("Sou coordenador!")
					self.publica_coor()
					for a in self.cluster_local[1:]:
						self.le_agente(a)
						print(self.agente_msg)
					if self.cloud_msg[0]==[1]:
						self.reconfigura_rede()
				else:
					print("Sou agente!")
					self.publica_agente()
					self.le_coordenador(self.cluster_local[0])
					print(self.coor_msg)
					if self.coor_msg is not None:
						if self.cloud_msg[0]==[1]:
							self.reconfigura_rede()

	def rede(self, id, tipo, rec): #configura ou reconfigura rede
		if self.cluster_local == []:
				print("Rede ainda não definida --iniciando configuração")
		else:
			print("Rede definida")
			#print(globals().keys())
			if self.uav_num == self.cluster_local[0]:
				print("Sou coordenador!")
				self.publica_coor(id, tipo, rec)
				for a in self.cluster_local[1:]:
					self.le_agente(a)
					print(self.agente_msg)
				for c in self.coordenadores:
					self.le_coordenador(c)
					print(self.coor_msg)
				if self.cloud_msg[1]==[1]:
					self.reconfigura_rede(id, tipo, rec)
				if self.cloud_msg[1]==[2]:
					if self.cloud_msg[3]==[self.uav_num]:
						self.pontos = self.cloud_msg[-1]
						print(self.pontos)
			else:
				print("Sou agente!")
				self.publica_agente(id, tipo, rec)
				self.le_coordenador(self.cluster_local[0])
				print(self.coor_msg)
				print(self.cloud_msg)
				if self.cloud_msg[1]==[2]:
						if self.cloud_msg[3]==[self.uav_num]:
							self.pontos = self.cloud_msg[-1]
							print(self.pontos)
				if self.coor_msg is not None:
					if self.cloud_msg[1]==[1]:
						self.reconfigura_rede(id, tipo, rec)

	def FLU2ENU(self, msg):

		FLU_x = msg.pose.position.x * math.cos(self.current_heading) - msg.pose.position.y * math.sin(self.current_heading)
		FLU_y = msg.pose.position.x * math.sin(self.current_heading) + msg.pose.position.y * math.cos(self.current_heading)
		FLU_z = msg.pose.position.z

		return FLU_x, FLU_y, FLU_z

	def set_target_position_callback(self, msg):
		print("Received New Position Task!")

		if msg.header.frame_id == 'base_link':
			'''
			BODY_FLU
			'''
			# For Body frame, we will use FLU (Forward, Left and Up)
			#           +Z     +X
			#            ^    ^
			#            |  /
			#            |/
			#  +Y <------body

			self.frame = "BODY"

			print("body FLU frame")

			ENU_X, ENU_Y, ENU_Z = self.FLU2ENU(msg)

			ENU_X = ENU_X + self.local_pose.pose.position.x
			ENU_Y = ENU_Y + self.local_pose.pose.position.y
			ENU_Z = ENU_Z + self.local_pose.pose.position.z

			self.cur_target_pose = self.construct_target(ENU_X,
														 ENU_Y,
														 ENU_Z,
														 self.current_heading)


		else:
			'''
			LOCAL_ENU
			'''
			# For world frame, we will use ENU (EAST, NORTH and UP)
			#     +Z     +Y
			#      ^    ^
			#      |  /
			#      |/
			#    world------> +X

			self.frame = "LOCAL_ENU"
			print("local ENU frame")

			self.cur_target_pose = self.construct_target(msg.pose.position.x,
														 msg.pose.position.y,
														 msg.pose.position.z,
														 self.current_heading)

	'''
	 Receive A Custom Activity
	 '''

	def custom_activity_callback(self, msg):

		print("Received Custom Activity:", msg.data)

		if msg.data == "LAND":
			print("LANDING!")
			self.state = "LAND"
			self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
														 self.local_pose.pose.position.y,
														 0.1,
														 self.current_heading)

		if msg.data == "HOVER":
			print("HOVERING!")
			self.state = "HOVER"
			self.hover()

		else:
			print("Received Custom Activity:", msg.data, "not supported yet!")

	def set_target_yaw_callback(self, msg):
		print("Received New Yaw Task!")

		yaw_deg = msg.data * math.pi / 180.0
		self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
													 self.local_pose.pose.position.y,
													 self.local_pose.pose.position.z,
													 yaw_deg)

	'''
	return yaw from current IMU
	'''
	def q2yaw(self, q):
		if isinstance(q, Quaternion):
			rotate_z_rad = q.yaw_pitch_roll[0]
		else:
			q_ = Quaternion(q.w, q.x, q.y, q.z)
			rotate_z_rad = q_.yaw_pitch_roll[0]

		return rotate_z_rad

	def arm(self):
		if self.armService(True):
			return True
		else:
			print("Vehicle arming failed!")
			return False

	def disarm(self):
		if self.armService(False):
			return True
		else:
			print("Vehicle disarming failed!")
			return False

	def offboard(self):
		if self.flightModeService(custom_mode='OFFBOARD'):
			return True
		else:
			print("Vechile Offboard failed")
			return False

	def hover(self, h):

		self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
													 self.local_pose.pose.position.y,
													 self.local_pose.pose.position.z,
													 self.current_heading)
	
	def fixa(self, status_decolagem, h):
		if status_decolagem==True:
			rospy.init_node("offboard_"+self.uav)
			#for i in range(10):
			while self.current_heading is None:
				if self.current_heading is not None:
					break 
				else:
					print("Aguardando inicializacao.")
					time.sleep(0.5)
				heading_inicio=self.current_heading
				self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)

				#print ("self.cur_target_pose:", self.cur_target_pose, type(self.cur_target_pose))
				# if self.image is not None:
				# 	cv2.imshow('camera0', self.image)
				# if cv2.waitKey(1) & 0xFF == ord('q'):
				# 	break

			for i in range(10):
				self.local_target_pub.publish(self.cur_target_pose)
				self.arm_state = self.arm()
				self.offboard_state = self.offboard()
				time.sleep(0.2)
				while self.local_pose.pose.position.z < self.takeoff_height:
					self.local_target_pub.publish(self.cur_target_pose)
					self.arm_state = self.arm()
					self.offboard_state = self.offboard()
					#time.sleep(0.2)
					print("to aqui")

			if self.takeoff_detection():
				print("Veiculo decolou!")

			else:
				print("Falha na decolagem!")
				return

			'''
			arma e decola a primeira vez
			'''
			if self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):
				self.local_target_pub.publish(self.cur_target_pose)
				self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
														 self.local_pose.pose.position.y,
														 h,
														 self.current_heading)
				self.local_target_pub.publish(self.cur_target_pose)
				dados_camera=self.camera_msg
				print(dados_camera)
				desx=self.destino_msg[0]
				desy=self.destino_msg[1]
				print(desx, desy)

		else:
			'''
			se mantem na posição
			'''
			if self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):
				self.local_target_pub.publish(self.cur_target_pose)
				self.cur_target_pose = self.construct_target(self.local_pose.pose.position.x,
														 self.local_pose.pose.position.y,
														 h,
														 self.current_heading)
				self.local_target_pub.publish(self.cur_target_pose)
				dados_camera=self.camera_msg
				print(dados_camera)
				desx=self.destino_msg[0]
				desy=self.destino_msg[1]
				print(desx, desy)

	def takeoff_detection(self):
		if self.local_pose.pose.position.z > 0.1 and self.offboard_state and self.arm_state:
			return True
		else:
			return False

	def navegue(self, status_decolagem,j=0):
		id = 0
		navx = []
		navy = []
		navz = []
		navheading = []
		navt = []
		if status_decolagem==True:
			self.rede(id, 9000, 9000) #9000 é um tipo genérico ou um receptor generico
			while self.current_heading is None:
				id = id+1
				self.rede(id, 9000, 9000)
				if self.current_heading is not None:
					break
				else:
					print("Aguardando inicializacao.")
					time.sleep(0.5)
				heading_inicio=self.current_heading
				self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)
				#self.controle_gimbal=self.controle_gimbal_camera(0, 0, 0)
				#self.controle_gimbal_pub.publish(self.controle_gimbal)
				#print ("self.cur_target_pose:", self.cur_target_pose, type(self.cur_target_pose))
				# if self.image is not None:
				# 	cv2.imshow('camera0', self.image)
				# if cv2.waitKey(1) & 0xFF == ord('q'):
				# 	break
				t=datetime.now()
				t=float(t.strftime("%Y%m%d%H%M%S.%f"))
				navx.append(self.local_pose.pose.position.x)
				navy.append(self.local_pose.pose.position.y)
				navz.append(self.local_pose.pose.position.z)
				navheading.append(self.current_heading)
				navt.append(t)

			for i in range(10):
				self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)
				self.local_target_pub.publish(self.cur_target_pose)
				self.arm_state = self.arm()
				self.offboard_state = self.offboard()
				time.sleep(0.2)
				disth = np.absolute(self.local_pose.pose.position.z - self.takeoff_height)
				while disth > 1:
					id = id+1
					self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)
					self.local_target_pub.publish(self.cur_target_pose)
					self.arm_state = self.arm()
					self.offboard_state = self.offboard()
					self.local_target_pub.publish(self.cur_target_pose)
					#self.controle_gimbal=self.controle_gimbal_camera(0, 0, 0)
					#self.controle_gimbal_pub.publish(self.controle_gimbal)
					#time.sleep(0.2)
					disth = np.absolute(self.local_pose.pose.position.z - self.takeoff_height)
					print("Decolando")
					self.rede(id, 9000, 9000)

			if self.takeoff_detection():
				print("Veiculo decolou!")

			else:
				print("Falha na decolagem!")
				return
		'''
		main ROS thread
		'''
		j=0
		while self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):
			#self.local_target_pub.publish(self.cur_target_pose)
			#self.controle_gimbal=self.controle_gimbal_camera(0, 0, 0)
			#self.controle_gimbal_pub.publish(self.controle_gimbal)
			if self.pontos == None:
				id = id+1
				self.rede(id, 9000, 9000)
				self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)
				self.local_target_pub.publish(self.cur_target_pose)
				self.arm_state = self.arm()
				self.offboard_state = self.offboard()
				self.local_target_pub.publish(self.cur_target_pose)
				print("Pontos de navegação não recebidos")
			else:
				id = id+1
				self.rede(id, 9000, 9000)
				if j < len(self.pontos):
					xt=self.pontos[j][0]
					yt=self.pontos[j][1]
					zt=self.pontos[j][2]
					#yawt=np.float32(pontos[j][3]*np.pi/180)
					yawt=np.arctan((self.local_pose.pose.position.y-yt)/(self.local_pose.pose.position.x-xt))
					self.cur_target_pose = self.construct_target(xt,yt,zt,yawt)
					self.local_target_pub.publish(self.cur_target_pose)
					x=self.local_pose.pose.position.x
					y=self.local_pose.pose.position.y
					z=self.local_pose.pose.position.z
					yaw=self.current_heading
					t=datetime.now()
					t=float(t.strftime("%Y%m%d%H%M%S.%f"))
					navx.append(x)
					navy.append(y)
					navz.append(z)
					navheading.append(yaw)
					navt.append(t)
					precisao = 0.5
					dist_tgt = np.sqrt(((x-xt)**2) + ((y-yt)**2) + ((z-zt)**2))
					print("UAV" + str(self.uav_num)+ " navegando para o ponto " + str(j) +"("+str(xt)+", "+str(yt)+", "+str(zt)+")")
					if dist_tgt <= precisao:
						j=j+1
						if j>=len(self.pontos):
							self.pontos = self.pontos[::-1]
							j=0
							if self.plotou == False:
								plt.plot(navx,navy)
								plt.xlabel("X (m)") 
								plt.ylabel("Y (m)")
								plt.title("Navigated trajectory of UAV "+str(self.uav_num))
								plt.grid()
								plt.savefig('/home/gabryelsr/src/Firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/missoes/dados/trajetoria_uav'+str(self.uav_num)+'.png')
								plt.clf()
								trajetoria = []
								for i in range(len(navx)): #matriz historico de trajetorias
									trajetoria.append([navx[i], navy[i], navz[i]])
								with open("/home/gabryelsr/src/Firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/missoes/dados/trajetoria_uav"+str(self.uav_num)+".json", 'w') as f:
									json.dump(trajetoria, f, indent=2)
								fig=plt.figure()
								ax = plt.axes(projection='3d')
								ax.plot3D(navx,navy,navz)
								ax.set_xlabel('X (m)')
								ax.set_ylabel('Y (m)')
								ax.set_zlabel('Z (m)')
								ax.set_title("Navigated 3D trajectory of UAV "+str(self.uav_num))
								ax.view_init(-140, 45)
								plt.savefig('/home/gabryelsr/src/Firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/missoes/dados/trajetoria3D_uav'+str(self.uav_num)+'.png')
								self.plotou = True
				if self.image is not None:
					cv2.imshow(self.frame_cam, self.image)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break
				if (self.state == "LAND") and (self.local_pose.pose.position.z < 0.15):
					if(self.disarm()):
						self.state = "DISARMED"
			time.sleep(0.1)
			#print(self.current_heading*180/np.pi)
			#self.publica_posicao()
		#salvar dataframe
		self.bd.to_excel('/home/gabryelsr/src/Firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/missoes/dados/banco_de_dados_uav'+str(self.uav_num)+'.xlsx')
		trajetoria = []
		for i in range(len(navx)): #matriz historico de trajetorias
			trajetoria.append([navx[i], navy[i], navz[i], navheading[i], navt[i]])
		with open("/home/gabryelsr/src/Firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/missoes/dados/banco_trajetoria_uav"+str(self.uav_num)+".json", 'w') as f:
			json.dump(trajetoria, f, indent=2)

	def camera(self):
		while self.image is not None:
			cv2.imshow(self.frame_cam, self.image)
			if cv2.waitKey(1) & 0xFF == ord('q'):
					break

	def missao_pt1(self, pontos,j=0):
	#rospy.init_node("offboard_node")
		for i in range(10):
			if self.current_heading is not None:
				break
			else:
				print("Aguardando inicializacao.")
				time.sleep(0.5)
		heading_inicio=self.current_heading
		self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)

		#print ("self.cur_target_pose:", self.cur_target_pose, type(self.cur_target_pose))

		for i in range(10):
			self.local_target_pub.publish(self.cur_target_pose)
			self.arm_state = self.arm()
			self.offboard_state = self.offboard()
			time.sleep(0.2)
			while self.local_pose.pose.position.z < self.takeoff_height:
				self.local_target_pub.publish(self.cur_target_pose)
				self.arm_state = self.arm()
				self.offboard_state = self.offboard()
				#time.sleep(0.2)
				print("to aqui")

		if self.takeoff_detection():
			print("Veiculo decolou!")

		else:
			print("Falha na decolagem!")
			return

		'''
		main ROS thread
		'''
		log_altura=[]
		log_yaw=[]
		log_tempo=[]
		log_x=[]
		log_y=[]
		while self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):
			self.local_target_pub.publish(self.cur_target_pose)
			if j < len(pontos):
				xt=pontos[j][0]
				yt=pontos[j][1]
				zt=pontos[j][2]
				yawt=np.float32(pontos[j][3]*np.pi/180)
				self.cur_target_pose = self.construct_target(xt,yt,zt,yawt)
				self.local_target_pub.publish(self.cur_target_pose)
				x=self.local_pose.pose.position.x
				y=self.local_pose.pose.position.y
				z=self.local_pose.pose.position.z
				yaw=self.current_heading
				t=rospy.get_time()
				log_altura.append(z)
				log_yaw.append(yaw*180/np.pi)
				log_tempo.append(t)
				log_x.append(x)
				log_y.append(y)
			if np.absolute(x-xt)<=2 and np.absolute(y-yt)<=2 and np.absolute(z-zt)<=2:
				j=j+1
				print(j)

			if j > len(pontos):
				log_etapa1={'X':log_x, 'Y':log_y, 'Altura':log_altura, 'Orientacao':log_yaw, 'Tempo':log_tempo}
				log_etapa1 = pd.DataFrame(data=log_etapa1)
				log_etapa1.to_csv('log_etapa1.csv', index=False)
				log_etapa1.to_csv('/home/gabryelcefetrj/src/Firmware/Tools/sitl_gazebo/missoes/log_pratico_etapa1.csv', index=False)
				plt.plot(log_tempo,log_altura)
				plt.xlabel('Tempo (s)')
				plt.ylabel('Altura (m)')
				plt.title('Altura x Tempo de Simulacao')
				plt.grid()
				plt.savefig('log_AlturaxTempo.png')

				plt.clf()
				plt.plot(log_tempo,log_yaw)
				plt.xlabel('Tempo (s)')
				plt.ylabel('Orientacao (º)')
				plt.title('Orientacao x Tempo de Simulacao')
				plt.grid()
				plt.savefig('log_YawxTempo.png')

				plt.clf()
				fig=plt.figure()
				ax = plt.axes(projection='3d')
				ax.plot3D(log_x,log_y,log_altura)
				ax.set_xlabel('X (m)')
				ax.set_ylabel('Y (m)')
				ax.set_zlabel('Altura (m)')
				ax.set_title('Orientacao x Tempo de Simulacao')
				fig.savefig('log_posicaoDrone_etapa1.png')

				break

			if (self.state == "LAND") and (self.local_pose.pose.position.z < 0.15):

					if(self.disarm()):

							self.state = "DISARMED"
			time.sleep(0.1)
			print(self.current_heading*180/np.pi)

	def missao_pt2(self, status_decolagem, h=15):
		vel=1.0
		max_dist=5
		#centro=img_x/2
		#tolerancia=0.1*img_x
		kp=1
		if status_decolagem==True:
			#rospy.init_node("offboard_node")
			for i in range(10):
				if self.current_heading is not None:
					break
				else:
					print("Aguardando inicializacao.")
					time.sleep(0.5)
				heading_inicio=self.current_heading
				self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)

				#print ("self.cur_target_pose:", self.cur_target_pose, type(self.cur_target_pose))

			for i in range(10):
				self.local_target_pub.publish(self.cur_target_pose)
				self.arm_state = self.arm()
				self.offboard_state = self.offboard()
				time.sleep(0.2)


			if self.takeoff_detection():
				print("Veiculo decolou!")
			else:
				print("Decolagem do veiculo falhou!")
				return

			'''
			decola e navega controlando velocidade a primeira vez
			'''
			obj_x0=0
			teta=self.current_heading*180/np.pi
			h=-34
			indicador=None
			log_altura=[]
			log_yaw=[]
			log_tempo=[]
			log_x=[]
			log_y=[]
			log_vel=[]

			primeira_deteccao=True
			contador=0
			yaw_compass=[]
			yaw_camera=[]
			yaw_kalman=[]
			ekf = DroneEKF()
			while self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):
				self.local_target_pub.publish(self.cur_target_pose)
				if self.camera_msg=='Parando Etapa 2!' or self.placa_msg=='Parando Etapa 2!':

					log_etapa2={'X':log_x, 'Y':log_y, 'Altura':log_altura, 'Orientacao':log_yaw, 'Tempo':log_tempo}
					log_etapa2 = pd.DataFrame(data=log_etapa2)
					log_etapa2.to_csv('log_etapa2.csv', index=False)
					log_etapa2.to_csv('/home/gabryelcefetrj/src/Firmware/Tools/sitl_gazebo/missoes/log_etapa2.csv', index=False)

					plt.clf()
					plt.plot(log_tempo,log_yaw)
					plt.xlabel('Tempo (s)')
					plt.ylabel('Orientacao (º)')
					plt.title('Orientacao x Tempo de Simulacao')
					plt.grid()
					plt.savefig('log_OrientacaoxTempo_etapa2.png')

					plt.clf()
					fig=plt.figure()
					ax = plt.axes(projection='3d')
					ax.plot3D(log_x,log_y,log_altura)
					ax.set_xlabel('X (m)')
					ax.set_ylabel('Y (m)')
					ax.set_zlabel('Altura (m)')
					ax.set_title('Orientacao x Tempo de Simulacao')
					fig.savefig('log_posicaoDrone_etapa2.png')

					break
				else:
					centrox=self.camera_msg[0]/2
					obj_x=self.camera_msg[2]
					tolerancia=0.2*centrox
					#print(teta)
					if obj_x0 != obj_x:
						if abs(centrox-obj_x)>tolerancia:
							if obj_x>centrox:
								print('Destino a direita')
								indicador='D'
								yaw_compass.append(teta)
								if primeira_deteccao == True:
									primeira_deteccao = False
									tetad=teta-kp*(abs(centrox-obj_x)/(2*centrox))
									if tetad<-180:
										tetad=tetad+360
								elif primeira_deteccao == False:
									tetad=yaw_camera[-1]-kp*(abs(centrox-obj_x)/(2*centrox))
									if tetad<-180:
										tetad=tetad+360
								yaw_camera.append(tetad)
								Tk=(yaw_compass[-1], yaw_camera[-1])
								dados_kalman=ekf.step(Tk)
								tetak=dados_kalman[0]
								yaw_kalman.append(tetak)
								contador=contador+1
								if contador>=10:
									tetaobj=tetak
								else:
									tetaobj=tetad
								print('######################################### NOVA ORIENTACAO:', tetaobj)
							else:
								print('Destino a esquerda')
								indicador='E'
								yaw_compass.append(teta)
								if teta>0:
									if primeira_deteccao == True:
										primeira_deteccao = False
										tetad=teta+kp*(abs(centrox-obj_x)/(2*centrox))
										if tetad>180:
											tetad=tetad-360
									elif primeira_deteccao == False:
										tetad=yaw_camera[-1]+kp*(abs(centrox-obj_x)/(2*centrox))
										if tetad>180:
											tetad=tetad-360
									yaw_camera.append(tetad)
								else:
									if primeira_deteccao == True:
										primeira_deteccao = False
										tetad=teta-kp*(abs(centrox-obj_x)/(2*centrox))
										if tetad<-180:
											tetad=tetad+360
									elif primeira_deteccao == False:
										tetad=yaw_camera[-1]-kp*(abs(centrox-obj_x)/(2*centrox))
										if tetad<-180:
											tetad=tetad+360
									yaw_camera.append(tetad)
								Tk=(yaw_compass[-1], yaw_camera[-1])
								dados_kalman=ekf.step(Tk)
								tetak=dados_kalman[0]
								yaw_kalman.append(tetak)
								contador=contador+1
								if contador>=10:
									tetaobj=tetak
								else:
									tetaobj=tetad
								print('######################################### NOVA ORIENTACAO:', tetad)
							while abs(tetaobj-self.current_heading*180/np.pi)>=2 or abs(h-self.local_pose.pose.position.z)>=0.1:
								self.cur_target_pose=self.controle_orientacao(tetaobj, h)
								self.local_target_pub.publish(self.cur_target_pose)
								time.sleep(0.2)
								teta=self.current_heading*180/np.pi
								print('Orientacao atual do drone:', teta)
								print('Orientacao desejada:', tetaobj)
								if indicador=='D':
									print('Corrigindo para a direita')
								else:
									print('Corrigindo para a esquerda')
					else:
						while abs(h-self.local_pose.pose.position.z)>=0.1:
							teta=self.current_heading*180/np.pi
							print('Orientacao atual do drone:', teta)
							self.cur_target_pose=self.controle_orientacao(teta, h)
							self.local_target_pub.publish(self.cur_target_pose)
							time.sleep(0.2)
						pass

					(orientacao, velx, vely) = self.calcula_velocidade_drone(vel, teta)
					self.velocidade_drone = self.controle_velocidade(velx, vely)
					self.setpoint_velocidade_pub.publish(self.velocidade_drone)
					print('Orientacao atual do drone:', teta)
					x=self.local_pose.pose.position.x
					xd=self.destino_msg[0]
					y=self.local_pose.pose.position.y
					yd=self.destino_msg[1]
					dist=np.sqrt(np.square(x-xd)+np.square(y-yd))
					print('Distancia do alvo:', dist)
					obj_x0=obj_x
					time.sleep(0.2)

					pos_z=self.local_pose.pose.position.z
					pos_x=self.local_pose.pose.position.x
					pos_y=self.local_pose.pose.position.y
					# vel_x=self.leitura_velocidade_drone.twist.linear.x
					# vel_y=self.leitura_velocidade_drone.twist.linear.y
					# vel_drone=np.sqrt((vel_x**2)+(vel_y**2))
					yaw_atual=self.current_heading*180/np.pi
					# ohm=self.leitura_velocidade_drone.twist.angular.z
					t=rospy.get_time()

					log_altura.append(pos_z)
					log_x.append(pos_x)
					log_y.append(pos_y)
					log_yaw.append(yaw_atual)
					# log_vel.append(vel_drone)
					log_tempo.append(t)

					if dist<=1:

						log_etapa2={'X':log_x, 'Y':log_y, 'Altura':log_altura, 'Orientacao':log_yaw, 'Tempo':log_tempo}
						log_etapa2 = pd.DataFrame(data=log_etapa2)
						log_etapa2.to_csv('log_etapa2.csv', index=False)
						log_etapa2.to_csv('/home/gabryelcefetrj/src/Firmware/Tools/sitl_gazebo/missoes/log_etapa2.csv', index=False)

						plt.clf()
						plt.plot(log_tempo,log_yaw)
						plt.xlabel('Tempo (s)')
						plt.ylabel('Orientacao (º)')
						plt.title('Orientacao x Tempo de Simulacao')
						plt.grid()
						plt.savefig('log_OrientacaoxTempo_etapa2.png')

						plt.clf()
						fig=plt.figure()
						ax = plt.axes(projection='3d')
						ax.plot3D(log_x,log_y,log_altura)
						ax.set_xlabel('X (m)')
						ax.set_ylabel('Y (m)')
						ax.set_zlabel('Altura (m)')
						ax.set_title('Orientacao x Tempo de Simulacao')
						fig.savefig('log_posicaoDrone_etapa2.png')

						break

		else:
			'''
			navega controlando velocidade
			'''
			obj_x0=0
			teta=self.current_heading*180/np.pi
			indicador=None
			h=-34
			log_altura=[]
			log_yaw=[]
			log_tempo=[]
			log_x=[]
			log_y=[]
			log_vel=[]

			primeira_deteccao=True
			contador=0
			yaw_compass=[]
			yaw_camera=[]
			yaw_kalman=[]
			ekf = DroneEKF()
			while self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):
				self.local_target_pub.publish(self.cur_target_pose)
				if self.camera_msg=='Parando Etapa 2!' or self.placa_msg=='Parando Etapa 2!':

					log_etapa2={'X':log_x, 'Y':log_y, 'Altura':log_altura, 'Orientacao':log_yaw, 'Tempo':log_tempo}
					log_etapa2 = pd.DataFrame(data=log_etapa2)
					log_etapa2.to_csv('log_etapa2.csv', index=False)
					log_etapa2.to_csv('/home/gabryelcefetrj/src/Firmware/Tools/sitl_gazebo/missoes/log_etapa2.csv', index=False)

					plt.clf()
					plt.plot(log_tempo,log_yaw)
					plt.xlabel('Tempo (s)')
					plt.ylabel('Orientacao (º)')
					plt.title('Orientacao x Tempo de Simulacao')
					plt.grid()
					plt.savefig('log_OrientacaoxTempo_etapa2.png')

					plt.clf()
					fig=plt.figure()
					ax = plt.axes(projection='3d')
					ax.plot3D(log_x,log_y,log_altura)
					ax.set_xlabel('X (m)')
					ax.set_ylabel('Y (m)')
					ax.set_zlabel('Altura (m)')
					ax.set_title('Orientacao x Tempo de Simulacao')
					fig.savefig('log_posicaoDrone_etapa2.png')

					break
				else:
					centrox=self.camera_msg[0]/2
					obj_x=self.camera_msg[2]
					tolerancia=0.2*centrox
					#print(teta)
					if obj_x0 != obj_x:
						if abs(centrox-obj_x)>tolerancia:
							if obj_x>centrox:
								print('Destino a direita')
								indicador='D'
								yaw_compass.append(teta)
								if primeira_deteccao == True:
									primeira_deteccao = False
									tetad=teta-kp*(abs(centrox-obj_x)/(2*centrox))
									if tetad<-180:
										tetad=tetad+360
								elif primeira_deteccao == False:
									tetad=yaw_camera[-1]-kp*(abs(centrox-obj_x)/(2*centrox))
									if tetad<-180:
										tetad=tetad+360
								yaw_camera.append(tetad)
								Tk=(yaw_compass[-1], yaw_camera[-1])
								dados_kalman=ekf.step(Tk)
								tetak=dados_kalman[0]
								yaw_kalman.append(tetak)
								contador=contador+1
								if contador>=10:
									tetaobj=tetak
								else:
									tetaobj=tetad
								print('######################################### NOVA ORIENTACAO:', tetaobj)
							else:
								print('Destino a esquerda')
								indicador='E'
								yaw_compass.append(teta)
								if teta>0:
									if primeira_deteccao == True:
										primeira_deteccao = False
										tetad=teta+kp*(abs(centrox-obj_x)/(2*centrox))
										if tetad>180:
											tetad=tetad-360
									elif primeira_deteccao == False:
										tetad=yaw_camera[-1]+kp*(abs(centrox-obj_x)/(2*centrox))
										if tetad>180:
											tetad=tetad-360
									yaw_camera.append(tetad)
								else:
									if primeira_deteccao == True:
										primeira_deteccao = False
										tetad=teta-kp*(abs(centrox-obj_x)/centrox)
										if tetad<-180:
											tetad=tetad+360
									elif primeira_deteccao == False:
										tetad=yaw_camera[-1]-kp*(abs(centrox-obj_x)/centrox)
										if tetad<-180:
											tetad=tetad+360
									yaw_camera.append(tetad)
								Tk=(yaw_compass[-1], yaw_camera[-1])
								dados_kalman=ekf.step(Tk)
								tetak=dados_kalman[0]
								yaw_kalman.append(tetak)
								contador=contador+1
								if contador>=10:
									tetaobj=tetak
								else:
									tetaobj=tetad
								print('######################################### NOVA ORIENTACAO:', tetad)
							while abs(tetaobj-self.current_heading*180/np.pi)>=2 or abs(h-self.local_pose.pose.position.z)>=0.1:
								self.cur_target_pose=self.controle_orientacao(tetaobj, h)
								self.local_target_pub.publish(self.cur_target_pose)
								time.sleep(0.2)
								teta=self.current_heading*180/np.pi
								print('Orientacao atual do drone:', teta)
								print('Orientacao desejada:', tetaobj)
								if indicador=='D':
									print('Corrigindo para a direita')
								else:
									print('Corrigindo para a esquerda')
					else:
						while abs(h-self.local_pose.pose.position.z)>=0.1:
							teta=self.current_heading*180/np.pi
							print('Orientacao atual do drone:', teta)
							self.cur_target_pose=self.controle_orientacao(teta, h)
							self.local_target_pub.publish(self.cur_target_pose)
							time.sleep(0.2)
						pass

					(orientacao, velx, vely) = self.calcula_velocidade_drone(vel, teta)
					self.velocidade_drone = self.controle_velocidade(velx, vely)
					self.setpoint_velocidade_pub.publish(self.velocidade_drone)
					print('Orientacao atual do drone:', teta)
					x=self.local_pose.pose.position.x
					xd=self.destino_msg[0]
					y=self.local_pose.pose.position.y
					yd=self.destino_msg[1]
					dist=np.sqrt(np.square(x-xd)+np.square(y-yd))
					print('Distancia do alvo:', dist)
					obj_x0=obj_x
					time.sleep(0.2)

					pos_z=self.local_pose.pose.position.z
					pos_x=self.local_pose.pose.position.x
					pos_y=self.local_pose.pose.position.y
					# vel_x=self.leitura_velocidade_drone.twist.linear.x
					# vel_y=self.leitura_velocidade_drone.twist.linear.y
					# vel_drone=np.sqrt((vel_x**2)+(vel_y**2))
					yaw_atual=self.current_heading*180/np.pi
					# ohm=self.leitura_velocidade_drone.twist.angular.z
					t=rospy.get_time()

					log_altura.append(pos_z)
					log_x.append(pos_x)
					log_y.append(pos_y)
					log_yaw.append(yaw_atual)
					# log_vel.append(vel_drone)
					log_tempo.append(t)

					if dist<=1:

						log_etapa2={'X':log_x, 'Y':log_y, 'Altura':log_altura, 'Orientacao':log_yaw, 'Tempo':log_tempo}
						log_etapa2 = pd.DataFrame(data=log_etapa2)
						log_etapa2.to_csv('log_etapa2.csv', index=False)
						log_etapa2.to_csv('/home/gabryelcefetrj/src/Firmware/Tools/sitl_gazebo/missoes/log_etapa2.csv', index=False)

						plt.clf()
						plt.plot(log_tempo,log_yaw)
						plt.xlabel('Tempo (s)')
						plt.ylabel('Orientacao (º)')
						plt.title('Orientacao x Tempo de Simulacao')
						plt.grid()
						plt.savefig('log_OrientacaoxTempo_etapa2.png')

						plt.clf()
						fig=plt.figure()
						ax = plt.axes(projection='3d')
						ax.plot3D(log_x,log_y,log_altura)
						ax.set_xlabel('X (m)')
						ax.set_ylabel('Y (m)')
						ax.set_zlabel('Altura (m)')
						ax.set_title('Orientacao x Tempo de Simulacao')
						fig.savefig('log_posicaoDrone_etapa2.png')

						break
	
	def publica_posicao(self):
		#rate = rospy.Rate(10)
		if not rospy.is_shutdown():
			self.current_pos_x = self.local_pose.pose.position.x
			self.current_pos_y = self.local_pose.pose.position.y
			self.current_pos_z = self.local_pose.pose.position.z
			log = [self.current_pos_x,
			       self.current_pos_y,
			       self.current_pos_z,
			       self.current_heading]
			log = str(log).strip('[]')
			rospy.loginfo(log)
			self.posicao_uav.publish(log)
			if self.cloud_msg is not None:
				print(self.cloud_msg[1])
			else:
				print(self.cloud_msg)
			#rate.sleep()
		
	def segue_alvo(self, status_decolagem):
		if status_decolagem==True:
			#rospy.init_node("offboard_uav0")
			#for i in range(10):
			while self.current_heading is None:
				
				if self.current_heading is not None:
					break
				else:
					print("Aguardando inicializacao.")
					time.sleep(0.5)

				#self.controle_gimbal=self.controle_gimbal_camera(0, 0, 0)
				#self.controle_gimbal_pub.publish(self.controle_gimbal)

				#print ("self.cur_target_pose:", self.cur_target_pose, type(self.cur_target_pose))
				# if self.image is not None:
				# 	cv2.imshow('camera0', self.image)
				# if cv2.waitKey(1) & 0xFF == ord('q'):
				# 	break

			while self.local_pose.pose.position.z < self.takeoff_height:
				self.cur_target_pose = self.construct_target(0, 0, self.takeoff_height, self.current_heading)
				self.local_target_pub.publish(self.cur_target_pose)
				self.arm_state = self.arm()
				self.offboard_state = self.offboard()
				self.local_target_pub.publish(self.cur_target_pose)
				time.sleep(0.2)

			if self.takeoff_detection():
				print("Veiculo decolou!")

			else:
				print("Falha na decolagem!")
				return
					
		offset_x = 1
		offset_y = 0
		while self.arm_state and self.offboard_state and (rospy.is_shutdown() is False):	
			if self.pos_lider is not None:
				posicao = self.pos_lider
			posicao=posicao.replace("[","")
			posicao=posicao.replace("]","")
			posicao = [float(item) for item in posicao.split(",")]
			xt=posicao[0] + offset_x	
			yt=posicao[1] + offset_y
			zt=posicao[2]
			yawt=posicao[3]*180/np.pi
			self.cur_target_pose = self.construct_target(xt,yt,zt,yawt)
			self.local_target_pub.publish(self.cur_target_pose)

centro_msg=None
def centro_callback(data):
	global centro_msg
	centro_msg=data.data

msg=None
def callback(data):
	global msg
	msg=data.data

cam_msg=None
def camera_callback(data):
	global cam_msg
	cam_msg=CvBridge.imgmsg_to_cv2(data)
   
placa_msg=None
def placa_callback(data):
	global placa_msg
	placa_msg=data.data

#rospy.init_node("controlador_offboard_iris0")
rospy.init_node("offboard_uav"+str(numero_uav))
if __name__ == '__main__':
	#DONT_RUN=1 make px4_sitl_default gazebo primeiro make apos o git clone
	#source Tools/simulation/gazebo/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default
	#export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)
	#export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)/Tools/simulation/gazebo/sitl_gazebo
	#roslaunch px4 posix_sitl.launch lanca 1 drone
	#roslaunch px4 multi_uav_mavros_sitl.launch lanca varios drones
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

	h=15
	v=1.0
	# while centro_msg is None:
	#     rospy.Subscriber("configura_centro", String, centro_callback)
	#     if centro_msg is not None:
	#         dados_centro=centro_msg
	# C = [float(coordenadas) for coordenadas in dados_centro.split(",")]
	# print(C[0])
	# print(C[1])

	#while msg is None:
	#    rospy.Subscriber("configura_trajetoria", String, callback)
	#    if msg is not None:
	#        dados=msg
	#dados=dados.replace("[","")
	#dados=dados.replace("]","")
	#dados = [float(item) for item in dados.split(",")]

	con = Px4Controller()
	#tam=int(len(dados)/3)
	#x=np.zeros(tam)
	#y=np.zeros(tam)
	#yaw=np.zeros(tam)
	#for i in range(tam):
	#    x[i]=dados[i]
	#    y[i]=dados[i+tam]
	#    yaw[i]=dados[i+2*tam]
	#x=x.tolist()
	#y=y.tolist()
	#yaw=yaw.tolist()
	#pontos=[]
	#for i in range(tam):
	#    pontos.append((x[i],y[i],h,np.float32(yaw[i])))
	#print(pontos)

	#executado_pt1=False
	#executado_pt2=False
	decola=True
	ang=0
	""" pontos1=[[4,4,h],[24,4,h],[24,8,h],[4,8,h],[4,12,h],[24,12,h],[24,16,h],
			[4,16,h],[4,20,h],[24,20,h],[24,24,h], [4,24,h]]
	pontos2=[[4,-4,h],[24,-4,h],[24,-8,h],[4,-8,h],[4,-12,h],[24,-12,h],[24,-16,h],
			[4,-16,h],[4,-20,h],[24,-20,h],[24,-24,h], [4,-24,h]]
	
	centrox = random.randint(0,200)
	centroy = random.randint(0,200)
	raio = random.randint(1,10)
	pontos = []
	y=np.arange(0, raio, raio/100).tolist()
	for i in range(len(yp)):
		x = np.sqrt((raio)**2-(y[i]-centroy)**2)+centrox
		pontos.append([x,y[i],h]) """

	while True:
		#h=5
		#con.fixa(decola, h)
		con.navegue(decola)
		#con.sitl()
		#con.publica_posicao()
		#con.segue_alvo(decola)

	#if executado_pt1 is False:
	#    con.missao_pt1(pontos)
	#    executado_pt1=True
	#    print('Fim da Etapa 1')
	#    decola=False
	#if executado_pt1 is True and executado_pt2 is False:
	#    con.missao_pt2(decola)
	#    executado_pt2=True
	#    print('Fim da Etapa 2')
	#    while executado_pt2 is True:
	#        h=1.5
	#        con.fixa(decola, -34.0)
		#decola=False
		# while cam_msg is None:
		#   if decola==True:
		#       con.fixa(decola)
		#       decola=False
		#   rospy.Subscriber('dados_camera_kf', String, camera_callback)
		#   if cam_msg is not None:
		#       dados_cam=cam_msg
		#       dados_cam=dados_cam.replace("[","")
		#       dados_cam=dados_cam.replace("]","")
		#       dados_cam = [float(item_cam) for item_cam in dados_cam.split(",")]
		#       #con.missao_pt2(v, dados_cam[0], dados_cam[2], decola)
		#       #print(dados_cam[0])
		#       #print(dados_cam[2])
		# cam_msg=None

