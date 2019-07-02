import cv2
import numpy as np
import dlib

def extract_index_nparray(nparray):
	index=None
	for num in nparray[0]:
		index=num
		break

	return index



img=cv2.imread("/home/chiranjeev/Desktop/face_swapping/bradely.jpeg")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mask=np.zeros_like(img_gray)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

faces=detector(img_gray)

for face in faces:
	landmarks=predictor(img_gray,face)
	landmarks_points=[]

	for n in range(0,68):
		x=landmarks.part(n).x
		y=landmarks.part(n).y
		landmarks_points.append((x,y))

		# cv2.circle(img,(x,y),3,(0,0,255),1)

	points=np.array(landmarks_points,np.int32)
	convexhull=cv2.convexHull(points)

	# cv2.polylines(img,[convexhull],True,(255,0,0),3)
	cv2.fillConvexPoly(mask,convexhull,255)

	face_image1=cv2.bitwise_and(img,img,mask=mask)

	#Delaunay traingulation
	rect=cv2.boundingRect(convexhull)
	subdiv=cv2.Subdiv2D(rect)
	subdiv.insert(landmarks_points)
	traingles=subdiv.getTriangleList()
	traingles=np.array(traingles,dtype=np.int32)

	indexes_triangles=[]

	for t in traingles:
		pt1=(t[0],t[1])
		pt2=(t[2],t[3])
		pt3=(t[4],t[5])

		##pt1

		#print("pt1=\n",pt1)
		index_pt1=np.where((points==pt1).all(axis=1))
		index_pt1=extract_index_nparray(index_pt1)
		#print("index_pt1\n",index_pt1)

		##pt2

		#print("pt2=\n",pt2)
		index_pt2=np.where((points==pt2).all(axis=1))
		index_pt2=extract_index_nparray(index_pt2)
		#print("index_pt2\n",index_pt2)
	
		##pt3

		#print("pt3=\n",pt3)
		index_pt3=np.where((points==pt3).all(axis=1))
		index_pt3=extract_index_nparray(index_pt3)
		#print("index_pt3\n",index_pt3)

		if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
			triangle=[index_pt1,index_pt2,index_pt3]
			indexes_triangles.append(triangle) 


		# cv2.circle(img,pt1,3,(0,255,0),-1)
		# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/created_image_marked_pt1_of_triangles.jpg",img)
	
		# cv2.circle(img,pt2,3,(15,29,130),2)
		# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/created_image_marked_pt2_of_triangles.jpg",img)

		# cv2.circle(img,pt3,3,(139,55,10),2)
		# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/created_image_marked_pt3_of_triangles.jpg",img)


		# cv2.line(img,pt1,pt2,(0,0,255),1)
		# cv2.line(img,pt2,pt3,(0,0,255),1)
		# cv2.line(img,pt3,pt1,(0,0,255),1)
	
	##################################
	#printing indexes
	# print(indexes_triangles)
	##################################

	# cv2.imshow("created_image",img)
	# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/created_image.jpg",img)
	#cv2.imshow("face_image",face_image1)
	#cv2.imshow("mask",mask)

########FACE-2################

img2=cv2.imread("/home/chiranjeev/Desktop/face_swapping/faces2.jpeg")
img2_gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

faces2=detector(img2_gray)

for face in faces2:
	landmarks2=predictor(img2_gray,face)
	landmarks_points2=[]

	for n in range(0,68):
		x=landmarks2.part(n).x
		y=landmarks2.part(n).y
		landmarks_points2.append((x,y))


		#cv2.circle(img2,(x,y),3,(0,255,0),-1)

	#drawing triangle on face2 same as face1

	#cv2.imshow("created_image2",img2)
	#cv2.imwrite("created_image2.jpg",img2)

	points2=np.array(landmarks_points2,np.int32)
	convexhull2=cv2.convexHull(points2)

lines_space_mask=np.zeros_like(img_gray)
lines_space_new_face=np.zeros_like(img2)

img2_new_face=np.zeros_like(img2,np.uint8)

for triangle_index in indexes_triangles:

	#########first face#########

	tr1_pt1=landmarks_points[triangle_index[0]]
	tr1_pt2=landmarks_points[triangle_index[1]]
	tr1_pt3=landmarks_points[triangle_index[2]]
	traingle1=np.array([tr1_pt1,tr1_pt2,tr1_pt3],np.int32)
	rect1=cv2.boundingRect(traingle1)
	(x1,y1,w1,h1)=rect1
	# cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,255,0),1)
	cropped_triangle1=img[y1:y1+h1,x1:x1+w1]

	cropped_tr1_mask=np.zeros((h1,w1),np.uint8)
	points1=np.array([[tr1_pt1[0]-x1,tr1_pt1[1]-y1],
					[tr1_pt2[0]-x1,tr1_pt2[1]-y1],
					[tr1_pt3[0]-x1,tr1_pt3[1]-y1]],np.int32)

	cv2.fillConvexPoly(cropped_tr1_mask,points1,255)
	cropped_triangle1=cv2.bitwise_and(cropped_triangle1,cropped_triangle1,mask=cropped_tr1_mask)

	#linespace

	# cv2.line(lines_space_mask,tr1_pt1,tr1_pt2,255)
	# cv2.line(lines_space_mask,tr1_pt2,tr1_pt3,255)
	# cv2.line(lines_space_mask,tr1_pt1,tr1_pt3,255)
	lines_sapce=cv2.bitwise_and(img,img,mask=lines_space_mask)
	#########second face#########

	tr2_pt1=landmarks_points2[triangle_index[0]]
	tr2_pt2=landmarks_points2[triangle_index[1]]
	tr2_pt3=landmarks_points2[triangle_index[2]]
	triangle2=np.array([tr2_pt1,tr2_pt2,tr2_pt3],np.int32)
	rect2=cv2.boundingRect(triangle2)
	(x2,y2,w2,h2)=rect2
	# cv2.rectangle(img2,(x2,y2),(x2+w2,y2+h2),(0,255,0,1))
	cropped_triangle2=img2[y2:y2+h2,x2:x2+w2]
	
	cropped_tr2_mask=np.zeros((h2,w2),np.uint8)
	points2=np.array([[tr2_pt1[0]-x2,tr2_pt1[1]-y2],
					[tr2_pt2[0]-x2,tr2_pt2[1]-y2],
					[tr2_pt3[0]-x2,tr2_pt3[1]-y2]],np.int32)

	cv2.fillConvexPoly(cropped_tr2_mask,points2,255)
	

	# cv2.line(img2,tr2_pt1,tr2_pt2,(0,0,255),2)
	# cv2.line(img2,tr2_pt2,tr2_pt3,(0,0,255),2)
	# cv2.line(img2,tr2_pt3,tr2_pt1,(0,0,255),2)

	#WARP TRAINGLES
	points1=np.float32(points1)
	points2=np.float32(points2)
	#it will tell how much to swap these two triangles
	M=cv2.getAffineTransform(points1,points2)
	#print(M)

	#warping triangle1 into triangle2
	warped_triangle=cv2.warpAffine(cropped_triangle1,M,(w2,h2))
	warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
	#break

	#Reconstruct destination face
	img2_new_face_rect_area=img2_new_face[y2:y2+h2,x2:x2+w2]
	
	img2_new_face_gray=cv2.cvtColor(img2_new_face_rect_area,cv2.COLOR_BGR2GRAY)
	_,background_mask=cv2.threshold(img2_new_face_gray,1,255,cv2.THRESH_BINARY_INV) #to put face
	background=cv2.bitwise_and(warped_triangle,warped_triangle,mask=background_mask)
	img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, background)
	img2_new_face[y2:y2+h2,x2:x2+w2]=img2_new_face_rect_area

#face_swapped	

img2_face_mask=np.zeros_like(img2_gray)
img2_head_mask=cv2.fillConvexPoly(img2_face_mask,convexhull2,255)
img2_face_mask=cv2.bitwise_not(img2_head_mask)

img2_head_noface=cv2.bitwise_and(img2,img2,mask=img2_face_mask)

result=cv2.add(img2_head_noface,img2_new_face)


(x,y,w,h)=cv2.boundingRect(convexhull2)
center_face2=(int((x+x+w)/2),int((y+y+h)/2))
seamlessclone=cv2.seamlessClone(result,img2,img2_head_mask,center_face2,cv2.MIXED_CLONE)

cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/seamlessclone_result.jpg",seamlessclone)

cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/final_swapping_result.jpg",result)


# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/background.jpg",background)

# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/img2_new_face_triangle_area.jpg",img2_new_face)


# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/wrapped_triangle.jpg",warped_triangle)

# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/cropped_tr1_mask.jpg",cropped_tr1_mask)
# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/cropped_tr2_mask.jpg",cropped_tr2_mask)

# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/cropped_tr1_seperated_triangle1_mask.jpg",cropped_triangle1)
# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/cropped_tr2_seperated_triangle2_mask.jpg",cropped_triangle2)

# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/cropped_single_triangle_on_img.jpg",cropped_triangle1)
# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/cropped_single_triangle_on_img2.jpg",cropped_triangle2)


#cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/single_triangle_on_img.jpg",img)
#cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/single_triangle_on__img2.jpg",img2)

# cv2.imshow("same_pts_on_img2_as_img1",img2)
# cv2.imwrite("/home/chiranjeev/Desktop/face_swapping/same_pts_on_img2_as_img1.jpg",img2)