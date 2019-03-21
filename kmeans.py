import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import cv2 #MAKE SURE IT ISNT VERSION 3 IM USING VERSION 2.4.13 THIS IS VERY IMPORTANT
import os
import random
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomTreesEmbedding

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

data_tr = []
data_te = []

n_clusters = 256
mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size = 50).fit(desc_sel_100k) #used minibatch to make it quick, performance is similar but speed is way better
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


#fit the forest
forest = RandomForestClassifier(n_estimators=100, max_depth = 16, min_samples_split = 3,max_features = None, criterion = 'entropy', n_jobs = -1).fit(data_tr, labels)
#make predictions
hs = forest.predict(data_te)
#whats the accuracy
(forest.score(data_te, labels))*100
#how many matches is that
sum(labels == hs)

#forest.decision_path(data_tr)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels, hs)
import matplotlib.pyplot as plt



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    #plt.text(0.5, 0.5, fontsize=12)
    plt.ylabel('True label',fontsize=12)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('kmeanscf.png',bbox_inches='tight')
    #plt.show()

plot_confusion_matrix(cm, desc_tr.keys())
