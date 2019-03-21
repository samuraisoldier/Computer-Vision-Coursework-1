import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import cv2 #MAKE SURE IT ISNT VERSION 3 IM USING VERSION 2.4.13 THIS IS VERY IMPORTANT
import os
import random
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV

############### need to make the features first ##################
folderName = 'Caltech_101/101_ObjectCategories';
classList = [dI for dI in os.listdir(folderName) if os.path.isdir(os.path.join(folderName,dI))]    #make list of classes from folder names

imgIdx = {}
imgSel = [15,15]

#lets make some labels
#labes has a list of lists, labels has been flattened
j = 0
labes = []
while j < len(classList):
    labes.append([j]*15)
    j = j + 1
labels  = [item for items in labes for item in items] #flatten the list of lists

##################lets make some SIFTs

desc_tr = {}
desc_te = {}
desc_sl = []

for nem in classList:
    #make the dictionaries for each class
    desc_tr[nem] = {}
    desc_te[nem] = {}
    subFolderName = folderName + '/' + nem
    imgList = os.listdir(subFolderName) #list of the images in this class
    imgIdx[nem] = np.random.permutation(len(imgList)) #make a random order of them
    imgIdx_tr = imgIdx[nem][0:imgSel[0]] #first 15 are the random training images
    imgIdx_te = imgIdx[nem][imgSel[1]:sum(imgSel)] #second 15 are the random test images

    i =0
    while i < len(imgIdx_tr):
        #make training data
        img = cv2.imread(subFolderName + '/' + imgList[imgIdx_tr[i]]) # read in an image
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # greyscale the ting
        sift = cv2.SIFT() # i think this line and the line after can go outside the big for loop but tbh it doesnt take long to run
        dense=cv2.FeatureDetector_create("Dense") #dense bc we want as many feature vecs as possible
        kp=dense.detect(gray) #ngl not sure what this does but i think it gets the features we need
        kp,des=sift.compute(gray,kp) #and then this one returns the features(kp) and locations(des)
        desc_tr[nem][i] = des # save the training des
        desc_sl.append(des) #make the one we pick 1000000 from for later
        #make test data
        img_te = cv2.imread(subFolderName + '/' + imgList[imgIdx_te[i]])
        gray_te= cv2.cvtColor(img_te,cv2.COLOR_BGR2GRAY)
        dense=cv2.FeatureDetector_create("Dense")
        kp_te=dense.detect(gray_te)
        kp_te,des_te=sift.compute(gray_te,kp_te)
        desc_te[nem][i] = des_te

        i = i + 1


#randomly select 100k SIFT descriptors for clustering
desc_sel = np.concatenate(desc_sl)
rand100k = random.sample(range(0,len(desc_sel)),100000)
desc_sel_100k = desc_sel[rand100k,:]
type(desc_sl)
type(desc_sel_100k)
n_clusters = 256
binsizes = [8192]
clusres = {}
bestclus = 0
bestsum = 0
bestacc = 0
bestoob = 0

mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size = 50).fit(desc_sel_100k) #used minibatch to make it quick, performance is similar
mbcenters = mbkmeans.cluster_centers_ #get our centers
print(n_clusters)
#make the histograms inni

data_tr = []
data_te = []
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


#fit the forest
forest = RandomForestClassifier(n_estimators=100, max_depth = 30, min_samples_split = 2,max_features = None, criterion = 'entropy', n_jobs = -1, oob_score = True).fit(data_tr, labels)
#make predictions
hs = forest.predict(data_te)
#whats the accuracy
fs = (forest.score(data_te, labels))*100
#how many matches is that
fsu = sum(labels == hs)
oobscre = forest.oob_score_
fs
#forest.oob_score_
sizeres = []
sizeres.append(fs)
sizeres.append(fsu)
sizeres.append(oobscre)
clusres[n_clusters] = sizeres




parameters = {'n_estimators':[10,20,50,70,80,100,250,500,1000],'max_depth': [5,10,15,20,30], 'min_samples_split':[2,3,4,5] }
#parameters = {'n_estimators':[100],'max_depth': [20],'min_samples_split':[2,3,4,5] }
rfc = RandomForestClassifier(n_estimators=100,max_depth = None, criterion = 'entropy', n_jobs = -1, oob_score = True)

#make predictions
hs = rfc.predict(data_te)
#whats the accuracy
(rfc.score(data_te, labels))*100
#how many matches is that
sum(labels == hs)



clf = GridSearchCV(rfc, parameters, cv = 3)
clf.fit(data_tr, labels)
clf.cv_results_
res = pd.DataFrame(clf.cv_results_)

res.to_csv('testfinal.csv')
'''
bestclus
bestsum
bestacc
bestoob
clusres
clusresdf = pd.DataFrame(clusres)
clusresdf.to_csv('clustest3.csv')
