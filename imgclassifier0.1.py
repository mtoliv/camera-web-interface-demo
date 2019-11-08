
''' Image classifier. Gets image from socket 01, sends result to socket 50 
Version 0.1 ? Alterações por: Roberto dia 08/08/2019, ctrl+f "alteração 1", "alteração 2", "alteração 3"
2019-08-07
'''

import numpy as np
import socket
import struct
import cv2
from config import *
from skimage import morphology
import datetime
import time
from matplotlib import pyplot as plt
import pickle
from sklearn.neural_network import MLPClassifier
import os.path,shutil, os
import sys
from threading import Thread, Lock
import queue


DISTANCE_BORDER_TO_ROI = 70 # Accept images when the circle is detected at this distance from the top
IMAGES_TO_SKIP = 0 # Skip this number of images after a valid image, so that no portions of circles are detected
HIDDEN_LAYER_SIZE = 20 # Neurons in the hidden layer of the MLP
img_global = None
status = None
lock = Lock()
#q = queue.Queue(maxsize=1)

''' Load image from socket. Returns  OpenCV image '''
def get_file(sock,id,size):
	fileName = 'media/images/log' 
	#len_str = sock.recv(4)  # Get file size
	#size = struct.unpack('!i', len_str)[0]
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
		print('Bytes received:', len(f_str),'missing:',size)    
	img = cv2.imdecode(np.asarray(bytearray(f_str)),1)
	#print(type(img))
	#print("SAVING MOST RECENT IMAGE")


	#Uncomment following lines to log images on /log/ folder
	"""
	date_string = datetime.datetime.now().strftime("%H_%M_%S_%f")[:-4]
	fileName = fileName + '/' + '%s' % (date_string) + '.png'
	cv2.imwrite(fileName, img)
	"""


	#cv2.imshow('img',img)
	return img

''' Get image from file '''
def get_greyscale(filename):
#    with open(filename, 'rb') as fp:
#        data = fp.read()
#    img = cv2.imdecode(np.asarray(bytearray(data)),1)
	
	img = cv2.imread(filename, 0)
	
	return img


''' Remove background, detect if image contains circle.
	Return image or None '''
def process_image(img):
	
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Mudar para o HSV para detetar o verde facilmente

	lower = np.array([18, 0, 0])     # Destacar a cor verde
	upper = np.array([60, 255, 255])

	mask = cv2.inRange(hsv, lower, upper)
	# cv2.imshow('Mask', mask)

	ret, mask = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)
	# cv2.imshow('Threshold', mask)

	binarized = np.where(mask > 0.1, 1, 0)
	processed = morphology.remove_small_objects(binarized.astype(bool), min_size=3000, connectivity=1).astype(int)
	mask_x, mask_y = np.where(processed == 0)
	mask[mask_x, mask_y] = 0
	# cv2.imshow('Final', mask)

	imgDividida = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	rows, cols = mask.shape[:3]

	# Detetar o circulo
	middle = int(cols / 2)  # Linha central
	for j in range(DISTANCE_BORDER_TO_ROI): # Check if the circle is close to the top of the image
		if mask[j, middle] > 0:
			# print('Object found')
			return mask, True

	return mask, False


''' Sum columns or rows of a matrix. ax = 0 sums rows, =1 sums along columns '''
def sumColumn(matrix,ax):
	return np.sum(matrix, axis=ax)  # axis=1 says "get the sum along the columns"


'''Normalize vector v to interval [0,1]'''
def normalize(v):
	max_value = v.max()
	min_value = v.min()
	if max_value>min_value :
			result = (v - min_value) / (max_value - min_value)
	return result


''' Classify an image as valid or error (to reject).
  Fits the model if [y] is given and train = 1.
  Returns 0 is valid, 1 if invalid and must be rejected '''
def classify_image(img,train,y = [0],img_path = [None],img_original = None): #Argumento img_path adicionado por Roberto, 08/08/2019
	rows, cols = img.shape[:2]
	#print('Number of rows (max):', rows)
	#print('Number of columns (max):', cols)


	#ALTERAÇÃO 1 por Roberto 08/08/2019: Garantir que dimensões da imagem satisfazem
	#A fazer: Adicionar condição para rows < 288 e cols > 384?
	if rows > 288:
		while rows > 288:
			img=np.delete(img,rows-1,0)
			rows, cols = img.shape[:2]

	if cols < 384:
		newcol=np.zeros((rows,1))
		while cols < 384:
			#print(img.shape[:2])
			#print(newcol.shape[:2])
			img=np.hstack((img,newcol))
			rows, cols = img.shape[:2]
	#FIM ALTERAÇÃO 1



	#ALTERAÇÕES Roberto - julho 2019
#    newrow=np.zeros(cols)
#    img=np.vstack([img,newrow])
#    img=np.delete(img,383,1)
#    img=np.delete(img,382,1)
#    rows, cols = img.shape[:2]
	#print('Number of rows (max):', rows)
	#print('Number of columns (max):', cols)
	# Fim ALTERAÇÕES Roberto - julho 2019

	hor = sumColumn(img,1) # Create horizontal and vertical histograms
	ver = sumColumn(img,0)
	h = np.concatenate([hor,ver]) # Join histograms
	hn = normalize(h) # Normalize to [0,1]

 #    print(hn.shape)
	filename = 'neuralmodel.dat'      # Read model from file if it exists
	iteration_counter = 0
	prediction =  [2]
	
	if os.path.isfile(filename):
		mlp = pickle.load(open(filename, 'rb'))
		if train:
			while prediction[0] != y[0]:
				print('Will update the model with the samples, performing two epochs, y=', y)
				mlp.partial_fit([hn],y)
				#mlp.partial_fit([hn],y) # Two epochs
				pickle.dump(mlp, open(filename, 'wb'))
				print('Will predict, calling the model to predict ')
				prediction = mlp.predict(hn.reshape(1,-1))
				print('Predicted: ',prediction)
				iteration_counter = iteration_counter + 1
				if iteration_counter > 10:
					break
	else:
		mlp = MLPClassifier(hidden_layer_sizes=(HIDDEN_LAYER_SIZE),max_iter=10,verbose=True, tol=0.001)
		print('New model created with ',HIDDEN_LAYER_SIZE,' neurons in the hidden layer.')
		print('Will train model with the data available.')
		x = [hn,hn]
		y = [0, 1]
		mlp.fit(x,y)
		pickle.dump(mlp, open(filename, 'wb'))
	print('Will predict, calling the model to predict ')
	prediction = mlp.predict(hn.reshape(1,-1))
	print('Predicted: ',prediction)

 
 #ALTERAÇÃO 2 POR Roberto, 08/08/2019
 #RAZÃO: Não é necessário voltar a guardar novo .png e novo chart.png quando o pretendido é apenas reclassificar?
 ###### Save to file in folder accepted or rejected ##############################
	#fileName='accepted'    
	#if prediction>0:
	#	fileName='rejected'
	#date_string = datetime.datetime.now().strftime("%H_%M_%S")
	#fileName = fileName+'/img' + '_%s' % (date_string) + '.png'
	#print('Will save to folder ',fileName)
	#cv2.imwrite(fileName, img)
 #### Save histograms too ########################################################
	#plt.clf()
	#plt.plot(hn)
	#plt.savefig(fileName+'chart.png')
 #    plt.show()
 #FIM ALTERAÇÃO 2

 #ALTERAÇÃO 3 por ROBERTO 08/08/2019
 #Se o pretendido for reclassificar (já actualizou o modelo neural/neuralmodel.dat nas linhas 152 as 159?):
	# move ficheiros (tanto .png como chart.png) para as pastas correspondentes, sem criar novo .png nem chart.png
	if train:
		if y[0]==1: 
			shutil.move(img_path,'media/images/rejected/')
			shutil.move(img_path+"chart.png",'media/images/rejected/')
			shutil.move(img_path+"color.png",'media/images/rejected/')
			print('Moved from %s',img_path)
			print('to .../rejected')
		elif y[0]==0:
			shutil.move(img_path,'media/images/accepted/')
			shutil.move(img_path+"chart.png",'media/images/accepted/')
			shutil.move(img_path+"color.png",'media/images/accepted/')
			print('Moved from %s',img_path)
			print('to .../accepted')
	 #Se prentendido for classificar nova imagem, executa pedaço de código comentado em ALTERAÇÃO 2
	else:
###### Save to file in folder accepted or rejected ##############################
		fileName='media/images/accepted'    
		if prediction>0:
			fileName='media/images/rejected'
		date_string = datetime.datetime.now().strftime("%H_%M_%S_%f")[:-4]
		fileName = fileName+'/img' + '_%s' % (date_string) + '.png'
		print('Will save to folder ',fileName)
		cv2.imwrite(fileName+'color.png', img_original)
		cv2.imwrite(fileName, img)
	#### Save histograms too ########################################################
		plt.clf()
		plt.plot(hn)
		plt.savefig(fileName+'chart.png')
 # FIM ALTERAÇÃo 3

	return prediction # No error detected
 



''' Get images from the socket and process them '''
def run(socket_address):

	s = socket.socket()
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)        
	s.bind(socket_address)
	s.listen(1)
	print("Socket open at port ",socket_address,'listening to the port.')

	print("Waiting for connection from clients.")
	#sc, info = s.accept()

	
	while True:
		sc, info = s.accept()
		ip, port = str(info[0]), str(info[1])
		#q = queue.Queue()
		print("Client connected:", info)  
		print("CREATING THREAD")
		Thread(target=client_thread, args=(sc, ip, port)).start()             
		"""mssg = sc.recv(4)  # Get file size
		message_type = struct.unpack('!i', mssg)[0]
		print('message is of type ' + str(message_type))
		if message_type == 2147483647:
			try:
				img = get_file(sc,1)  # Get image from socket
				sc.sendall(OK_COMMAND.encode())
			finally:
				print('Received, waiting more.')
	#            sc.close()
	#            print("Waiting for connection from clients.")
	#            sc, info = s.accept()
			if last_image>0: # Skip a number of images after the last valid image
				last_image -= 1
				continue
			mask, to_process = process_image(img) # Process image
			if to_process: # Process this image, skip other images of the same object
				last_image = IMAGES_TO_SKIP
				valid = classify_image(mask,train = 0, y=[0])
		elif message_type == 2147483646:
			print('TO IMPLEMENT')
			print('TO IMPLEMENT')
			print('TO IMPLEMENT')
			print('TO IMPLEMENT')
			print('TO IMPLEMENT')
			print('TO IMPLEMENT')
			response = struct.pack('!i', 112)
			sc.sendall(response)"""
  

#    get_file(s,2)    
##    s.sendall(QUIT_COMMAND.encode())
#    s.sendall(OK_COMMAND.encode())
#
	print("Closing socket and exit")
	s.close()

def client_thread(sc,ip,port):#,q):
	global img_global, status
	last_image = 0  
	while True:
		mssg = sc.recv(4)  # Get file size
		"""try:
			print("received data:", mssg.decode())
			message_type = mssg.decode()
			#print("test")
			#print("testtest")
			#prin("1")
		except:
			message_type = struct.unpack('!i', mssg)[0]
			#print("test2")
			print(message_type)
			#print("2")
		print('message is of type ' + str(message_type))"""
		try:
			message = mssg.decode()
			size = struct.unpack('!i', mssg)[0]
			#size = None
		except:
			#message_type = struct.unpack('!i', mssg)[0]
			size = struct.unpack('!i', mssg)[0]
			message = "none"
		if message == "get_":
			#f = os.listdir(fileName)
			print("STR message")
			print(message)
			print('TO IMPLEMENT')  
			print('TO IMPLEMENT')
			print('TO IMPLEMENT')
			print('TO IMPLEMENT')
			print('TO IMPLEMENT')
			print('TO IMPLEMENT')
			#response = struct.pack('!i', 112)
			#sc.sendall(response)
			#if q.empty():
			#	print('q is empty')
			#n_img = q.get()
			#rows, cols = n_img.shape[:2]
			print('\nLOCKING\n')
			lock.acquire()
			rows, cols = img_global.shape[:2]
			print('Number of rows (max):', rows)
			print('Number of columns (max):', cols)
			encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
			result, frame = cv2.imencode('.png', img_global, encode_param)
			#lock.release()
			print('\nRELEASED\n')
			data = pickle.dumps(frame, 0)
			sc.sendall(struct.pack('!i', status))
			size = len(data)
			#len_str = struct.pack('!i', len(img_global))     # send string size                
			sc.sendall(struct.pack('!i', size))
			print('Sent filesize size to client: ',size,'(',size,')')
			# send bytes to socket
			sc.sendall(data)
			print('Sent data, will wait.')
			#sc.sendall(struct.pack('!i', status))
			rec_data = sc.recv(50)
			print('Received from server: ',rec_data.decode())
			lock.release()
			break
		else:
			print("INT message")
			print(size)
			try:
				img = get_file(sc,1,size)  # Get image from socket
				sc.sendall(OK_COMMAND.encode())
			finally:
				print('Received, waiting more.')
		#            sc.close()
		#            print("Waiting for connection from clients.")
		#            sc, info = s.accept()
			if last_image>0: # Skip a number of images after the last valid image
				last_image -= 1
				continue
			mask, to_process = process_image(img) # Process image
			if to_process: # Process this image, skip other images of the same object
				last_image = IMAGES_TO_SKIP
				valid = classify_image(mask,train = 0, y=[0],img_original = img) # 0 is accepted, 1 is rejected
				print("SAVING MOST RECENT IMAGE")
				fileName = 'media/images/recent'# + str(valid[0]) + '.png' 
				#os.remove(fileName)
				#os.rmdir(fileName)
				#os.mkdir(fileName)
				#files = os.listdir(fileName)
				#print(files)
				#while len(files) > 0:

				"""
				for file in os.listdir(fileName):
					string = fileName + '/' + file
					os.remove(string)
				date_string = datetime.datetime.now().strftime("%H_%M_%S_%f")[:-4]
				fileName = fileName + '/' + '%s' % (date_string) + '_' + str(valid[0]) + '.png'
				cv2.imwrite(fileName,img)
				"""

				#q.put(img)
				#print(q.empty())
				print('\nLOCKING\n')
				lock.acquire()
				img_global = img
				status = valid[0]
				lock.release()
				print('\nRELEASED\n')
		
# --- main ---

print('\nImage classifier server. If no arguments are given, will listen to socket.')
print('To classify an image, type:  \"img_classifier imagename\".')
print('To learn an image, type:  \"img_classifier imagename class\", where class is 1 to reject, 0 to accept.')


#inputfile='img/to_accept.png'
#mask, to_process = process_image(img) # Process image    
#cv2.imshow('Threshold', img)
#    cv2.waitKey(0)
#y = [ 0 ]
#print('Will learn image ',inputfile,' as ',y)
#out = classify_image(img,1,y)


if len(sys.argv)>1: # Get file name from input, if input is given
	inputfile = sys.argv[1]
	img = get_greyscale(inputfile)
	
#    cv2.imshow('Threshold', img)
#    cv2.waitKey(0)

	if len(sys.argv)>2: # Get file name from input, if input is given
		y = [ int(sys.argv[2]) ]
		print('Will learn image ',inputfile,' as ',y)
		out = classify_image(img,1,y,inputfile)
	else:
		print('Will classify image ',inputfile)
		out = classify_image(img,0)
	print('Prediction Output = ',out)
#    classify_image(mask,train = 0, y=[0])
else:
	run(CAMERA_ADDRESS)
