#!/usr/bin/env python3

import socket, os
import struct # to send `int` as  `4 bytes`
import time, datetime
from config import *
import cv2, pickle
import numpy as np
#FILENAME = 'chart.png'  # File to send


def get_file(filename):
	with open(filename, 'rb') as fp:
		data = fp.read()
	return data


''' Send image to socket '''
def send_image(sock):

#    sock.sendall(GET_IMAGE.encode())
#    data = sock.recv(50)
#    print('Received from server: ',data.decode())

#    while data.decode() != QUIT_COMMAND: # Loop - send images until receives QUIT_COMMAND
	#message_type = struct.pack('!i', 2147483646)
	message = "get_"
	sock.sendall(message.encode())
	#response = sock.recv(4)  # Get file size
	#response = struct.unpack('!i', response)[0]
	#print('server sent back' + str(response))
	"""fdata = get_file(FILENAME)
	print('Opened file of length:', len(fdata))
	len_str = struct.pack('!i', len(fdata))     # send string size                
	sock.sendall(len_str)
	print('Sent filesize size to client: ',len_str,'(',len(fdata),')')
	# send bytes to socket
	sock.sendall(fdata)
	print('Sent data, will wait.')
	data = sock.recv(50)
	print('Received from server: ',data.decode())"""
	status = sock.recv(4)
	status = struct.unpack('!i', status)[0]
	len_str = sock.recv(4)  # Get file size
	size = struct.unpack('!i', len_str)[0]
#    print('File size received:', size)
	f_str = b''
	while size > 0:
		if size >= 4096:
			data = sock.recv(4096)
		else:
			data = sock.recv(size)
			if not data:
				break                    
		size -= len(data)
		f_str += data
		#print('Bytes received:', len(f_str),'missing:',size)   
	frame=pickle.loads(f_str, fix_imports=True, encoding="bytes")
	#print(type(frame))
	#img = cv2.imdecode(np.asarray(bytearray(f_str)),1)
	img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
	#print(type(img))
	rows, cols = img.shape[:2]
	#print('Number of rows (max):', rows)
	#print('Number of columns (max):', cols)
	date_string = datetime.datetime.now().strftime("%H_%M_%S")
	fileName = 'media/images/recent'
	for file in os.listdir(fileName):
		string = fileName + '/' + file
		os.remove(string)
	#fileName = fileName + '/' + '%s' % (date_string) + '.png'
	#cv2.imwrite(fileName,img)
	#status = sock.recv(4)
	#status = struct.unpack('!i', status)[0]
	print('status is' + str(status))
	#fileName = fileName + '/' + '%s' % (date_string) + 'accepted.png'
	if status:
		fileName = fileName + '/' + '%s' % (date_string) + '_1.png'
	else:
		fileName = fileName + '/' + '%s' % (date_string) + '_0.png'
	cv2.imwrite(fileName,img)
	#cv2.imshow('ImageWindow',frame)
	#cv2.waitKey(1000)
	sock.sendall(OK_COMMAND.encode())


''' Open socket and send image to server '''
def run(socket_address):
	s = socket.socket()
	s.connect(socket_address)
	send_image(s)

	#time.sleep(0.5)


	# exit*
	print("Closing socket, will exit.")
	
	s.close()

# --- main ---

run(CAMERA_ADDRESS)