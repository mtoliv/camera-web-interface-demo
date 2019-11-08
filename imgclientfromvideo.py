#!/usr/bin/env python3

''' File to simulate a camera sending images to socket, for image classifier
Version 0.0
2019-07-12 '''

import socket
import struct # to send `int` as  `4 bytes`
from config import *
import cv2
import numpy as np
from skimage import morphology
import datetime
from time import sleep
import time, os
from random import shuffle
files = os.listdir('imgs/')
i=0
while i < 1:
	#print("everyday i'm shuffling")
	shuffle(files)
	print('.')
	i=i+0.01
	#time.sleep(0.1)

''' Open socket and send images to there '''
def run(socket_address):


	s = socket.socket()

	s.connect(socket_address)


	
	count = 1
	for file in files:
		print(file)
		with open('imgs/' + str(file), 'rb') as fp:
			fdata = fp.read()
		print('[',count,']Got file of length:', len(fdata))
		len_str = struct.pack('!i', len(fdata))     # send string size                
		s.sendall(len_str)
		print('Sent filesize size to client: ',len_str,'(',len(fdata),')')
		# send bytes to socket
		s.sendall(fdata)
		print('Sent data, will wait.')
		data = s.recv(50)
		print('Received from client: ',data.decode())
		count+=1
		time.sleep(2.5)
	# exit*
	print("Closing socket, will exit.")
#    time.sleep(0.5)
	s.close()

run(CAMERA_ADDRESS)
	
#count = 0
#while True:
#    data = get_file('x')
#    print('Frame',count,'length ',len(data))
##    cv2.imshow('data', data)
#    count+=1
#    if count>100: break
