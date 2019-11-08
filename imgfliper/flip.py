import cv2, os
  
'''originalImage = cv2.imread('C:/Users/N/Desktop/Test.jpg')
  
flipVertical = cv2.flip(originalImage, 0)
flipHorizontal = cv2.flip(originalImage, 1)
flipBoth = cv2.flip(originalImage, -1)'''
 

files = os.listdir('C:\\Users\\Roberto.Oliveira\\Desktop\\camera-web-interface_offline\\webinterface_windows\\webinterface\\imgfliper\\flip')
for fileName in files:
	#print(file)
	#file = cv2.imread(str(file),cv2.IMREAD_COLOR)
	#file = cv2.imread('img_20_48_34_46.pngcolor.png', cv2.IMREAD_COLOR)
	file = cv2.imread(str(fileName), cv2.IMREAD_COLOR)
	#print(file)
	flipVertical = cv2.flip(file, 0)
	flipHorizontal = cv2.flip(file, 1)
	flipBoth = cv2.flip(file, -1)

	#cv2.imwrite(str(file) + ', img)
	cv2.imwrite(str(fileName) + 'vertical.png', flipVertical)
	cv2.imwrite(str(fileName) + 'horizontal.png', flipHorizontal)
	cv2.imwrite(str(fileName) + 'both.png', flipBoth)
	print('hey')
	'''cv2.imshow('Flipped vertical image', flipVertical)
	cv2.imshow('Flipped horizontal image', flipHorizontal)
	cv2.imshow('Flipped both image', flipBoth)'''
	#break

	 
	 
#cv2.waitKey(0)
#cv2.destroyAllWindows()