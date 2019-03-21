import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import cv2 #MAKE SURE IT ISNT VERSION 3 IM USING VERSION 2.4.13 THIS IS VERY IMPORTANT
import os
import random
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import ExtraTreesClassifier

############### need to make the features first ##################
folderName = 'Caltech_101/101_ObjectCategories';
classList = [dI for dI in os.listdir(folderName) if os.path.isdir(os.path.join(folderName,dI))]    #make list of classes from folder names

imgIdx = {}


##################lets make some SIFTs




nem = 'wild_cat'
subFolderName = folderName + '/' + nem
imgList = os.listdir(subFolderName) #list of the images in this class
img = cv2.imread(subFolderName + '/' + imgList[4]) # read in an image
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # greyscale the ting
sift = cv2.SIFT() # i think this line and the line after can go outside the big for loop but tbh it doesnt take long to run
#dense=cv2.FeatureDetector_create("Dense") #dense bc we want as many feature vecs as possible
kp=sift.detect(gray) #ngl not sure what this does but i think it gets the features we need
kp,des=sift.compute(gray,kp) #and then this one returns the features(kp) and locations(des)
img2 = cv2.drawKeypoints(gray,kp, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints_wild_cat.jpg', img2)


############make histograms





nem = 'wheelchair'
subFolderName = folderName + '/' + nem
imgList = os.listdir(subFolderName) #list of the images in this class
img = cv2.imread(subFolderName + '/' + imgList[2]) # read in an image
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # greyscale the ting
sift = cv2.SIFT() # i think this line and the line after can go outside the big for loop but tbh it doesnt take long to run
#dense=cv2.FeatureDetector_create("Dense") #dense bc we want as many feature vecs as possible
kp=sift.detect(gray) #ngl not sure what this does but i think it gets the features we need
kp,des=sift.compute(gray,kp) #and then this one returns the features(kp) and locations(des)
img2 = cv2.drawKeypoints(gray,kp, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints_wild_cat.jpg', img2)

n_clusters = 
mbkmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(desc_sel_100k) #used minibatch to make it quick, performance is similar but speed is way better
mbcenters = mbkmeans.cluster_centers_ #get our centers
neigh = NearestNeighbors(1).fit(mbcenters) #fit our knn to the clusters from kmeans

#this loop takes the lonest idk why maybe bincount is slow
for nem in desc_tr.keys(): #keys are the same for training and test
    i =0
    while i < len(desc_tr[nem]):
        #training data
        this_col = desc_tr[nem][i]  #get the image we want
        hp = neigh.kneighbors(this_col) #get its nearest neighbors
        hp2 =np.array(list(hp[1].flatten()))
        hist = np.bincount(hp2, None, n_clusters) #make a histogram
        histlist = list(hist)
        data_tr.append(histlist)

        #test data
        this_img = desc_te[nem][i]
        hp_te = neigh.kneighbors(this_img)
        hp2_te =np.array(list(hp_te[1].flatten()))
        hist_te = np.bincount(hp2_te, None, n_clusters)
        histlist_te = list(hist_te)
        data_te.append(histlist_te)

        i = i + 1
