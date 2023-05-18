from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage, default_storage


# import cv2 as cv

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

# Create your views here.
#takes request and sends response 
#request handler
def upload_file(request):
    if request.method == "POST":
        my_uploaded_file = request.FILES['my_uploaded_file'].read() # get the uploaded file
        # do something with the file
        
        # and return the result  
        context={'filePathName':my_uploaded_file}
        return render(request,'index1.html',context)          
    else:
    	return render(request, 'index.html')


def read_File(request):
	return render(request, 'index.html', {'name': ''})

def crop_image(img):

	largest_cnt = find_max_contour(img)
	# Create bounding box around contour
	x, y, w, h = cv2.boundingRect(largest_cnt)

	# Crop image to bounding box
	cropped = img[y:y+h, x:x+w]
	gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
	return gray

def circle_contour_coordinates(contour):
	(x,y), radius = cv2.minEnclosingCircle(contour)
	center = (int(x),int(y))
	radius = int(radius)
	return center, radius

def find_max_contour(img):

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# Find contours
	contours, r = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Find contour with the largest area (assumed to be the sun)
	largest_contour = max(contours, key=cv2.contourArea)
	return contours, largest_contour

def draw_circle(img, x, y, radius):
	center = (int(x),int(y))
	radius = int(radius)

	# Draw a circle around the solar disk
	output_img =img.copy()
	cv2.circle(output_img, center, radius, (0,255,0), 2)

	# Display the output image
	return output_img

# def sobel_filter(image):
# 	context = {'a':1}
# 	return render(request, 'index.html', context)
	
def dbscanAlgo(request):
	print(request)
	print(request.POST.dict())
	print(request.FILES["filePath"])
	fileObj = request.FILES["filePath"]
	fs = FileSystemStorage()
	filePathName = fs.save(fileObj.name, fileObj)
	filePathName = fs.url(filePathName)
	print(filePathName)
	img = cv2.imread("C:/Users/dell/OneDrive/Desktop/WebApp"+filePathName)
	select1 = request.POST['select1']
	print(select1)
	input1 = request.POST['input2']
	print(input1)
	input2 = request.POST['input3']
	print(input2)
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Threshold the image
	m, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# Find contours
	contours, r = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Find contour with the largest area (assumed to be the sun)
	cnt = max(contours, key=cv2.contourArea)

	# Create bounding box around contour
	x, y, w, h = cv2.boundingRect(cnt)

	# Crop image to bounding box
	cropped = gray[y:y+h, x:x+w]

	image = cropped.copy()
	# Apply Gaussian blur to reduce noise (optional)
	image_blur = cv2.GaussianBlur(image, (5, 5), 0)
	# image_blur = 
	# Apply Sobel edge detection
	sobel_x = cv2.Sobel(image_blur, cv2.CV_64F, 1, 0, ksize=29)
	sobel_y = cv2.Sobel(image_blur, cv2.CV_64F, 0, 1, ksize=29)
	gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

	# Normalize the gradient magnitude
	gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
	b  = cv2.resize(gradient_magnitude_normalized, (700, 700))

	# Threshold the gradient magnitude to create a binary image
	threshold_value = 60  # Adjust this value to control the sensitivity
	_, binary_image = cv2.threshold(gradient_magnitude_normalized, threshold_value, 255, cv2.THRESH_BINARY)

	b  = cv2.resize(binary_image, (700, 700))

	# cv2.imshow("binary", b)
	# Find contours of sunspots
	contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Draw the contours on the original image
	image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)  # Red color for contours
	sunspots = []
	for contour in contours:
	    area = cv2.contourArea(contour)
	    if area <1000  and area>3:
	        # Calculate the centroid of the contour
	        moments = cv2.moments(contour)
	        cx = int(moments['m10'] / moments['m00'])
	        cy = int(moments['m01'] / moments['m00'])
	        sunspots.append((cx, cy, area))

	# Convert the list of sunspots to a numpy array
	sunspots = np.array(sunspots)
	
	# kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(sunspots[:, :2])
	dbscan = DBSCAN(eps=120, min_samples=1)

	# Run DBSCAN clustering
	dbscan.fit(sunspots[:, :2])

	# Get the cluster labels assigned to each data point
	labels = dbscan.labels_
	print('labels: ', labels)
	# # Draw a circle around each detected sunspot and a number indicating its label
	colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 250, 0), (25, 150, 100)]  # colors for each cluster

	l=0
	num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	print("Sunspot number: ", len(labels))
	print("Estimated number of clusters:", num_clusters)
	print("wolf number : ", len(labels)+ num_clusters*10)
	dist = sunspots[:, :2]
	areas = sunspots[:, 2]
	for i in range(num_clusters):
	    points_of_cluster = dist[labels==i,:]
	    centroid_of_cluster = np.mean(points_of_cluster, axis=0)
	    cluster_area = np.mean(areas[labels==i])
	    color = colors[l%5]
	    cv2.circle(cropped, (int(centroid_of_cluster[0]), int(centroid_of_cluster[1])), int(cluster_area*0.6), color, 2)
	    l+=1

	img2 = cv2.resize(cropped, (800,800))


	# Filename
	filename = './media/savedImage.jpg'
	  
	# Using cv2.imwrite() method
	# Saving the image
	cv2.imwrite(filename, img2)
	context = {'a':1, 'filePathName':filePathName, 'filename': filename, 'Sunspot': len(labels), 'num_clusters': num_clusters, 'wolf': num_clusters*10}
	return render(request, 'index.html', context)