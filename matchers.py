import cv2
import numpy as np 

class matchers:
	def __init__(self):
                # Initiate SURF detector
		self.surf = cv2.xfeatures2d.SURF_create()# capture more key points than SIFT
		FLANN_INDEX_KDTREE = 0# (Fast Library for Approximate Nearest Neighbors) to match key points
		index_params = dict(algorithm=0, trees=5)
		search_params = dict(checks=50)
		self.flann = cv2.FlannBasedMatcher(index_params, search_params)

	def match(self, i1, i2, direction=None):
		imageSet1 = self.getSURFFeatures(i1)
		imageSet2 = self.getSURFFeatures(i2)
		print ("Direction : ", direction)# left or right
		matches = self.flann.knnMatch(imageSet2['descriptor'], imageSet1['descriptor'], k=2)# use knn algrithom to match image
		# Apply ratio test
		good = []
		for i , (m, n) in enumerate(matches):
			if m.distance < 0.75*n.distance:
				good.append((m.trainIdx, m.queryIdx))

		if len(good) > 4:# minimum match count
			pointsCurrent = imageSet2['key points']
			pointsPrevious = imageSet1['key points']

			matchedPointsCurrent = np.float32(
				[pointsCurrent[i].pt for (__, i) in good]
			)
			matchedPointsPrev = np.float32(
				[pointsPrevious[i].pt for (i, __) in good]
				)

			H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
			return H # return transformation
		return None

	def getSURFFeatures(self, image):# get key points and descriptor
		# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)# change to gray model
		# find the key points and descriptors with SURF
		kp, des = self.surf.detectAndCompute(gray, None)
		return {'key points':kp, 'descriptor':des}
