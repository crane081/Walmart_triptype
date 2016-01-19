# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:59:49 2015

This code is for Kaggle challenge: Walmart trip type prediction
https://www.kaggle.com/c/walmart-recruiting-trip-type-classification

Data can be downloaded here:
https://www.kaggle.com/c/walmart-recruiting-trip-type-classification/data

@author: Crane Huang
"""
#%% hypothesis: features that may affect Triptype:
#1. weekday -> small daily dinner trip (Mon-Fri), weekly large grocery trip (weekends)
#small daily trip/large grocery trip: measure by items purchased (count each visitnumber, and may
#be SanCount)
#2. #items purchased
#3. Department -> holiday or seasonal trip
#4. if anything was returned
#5. if visited certain department 
#6. if purchased multiple quantites 

#goal: predict target (trip type, sorted by visitnumber), using:

#X_day: which weekday (0-6 for Mon-Sun)
#N_item: number of items purchased
#Return (1/0): if anything is returned in this trip
#N_dpt: number of departments visited
#Multi: if multiple quantities are purchased for at least 1 item
#Each_dept: if each department is visited (based on unique(Department)) (69)
#Tarshop: if at least half of the items were purchased from the same department (targeted shopping)

#y: trip type for each visit (38). 

#%% import libraries

import warnings
warnings.filterwarnings("ignore") #ignore all warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pdb import set_trace as bp

#%% import training data

df=pd.read_csv('train.csv')
y=df.TripType
VisitNumber0=df.VisitNumber
FinelineNumber=df.FinelineNumber
ScanCount=df.ScanCount
Visit=np.unique(VisitNumber0)
N_y=len(np.unique(y))
N_visit=len(Visit)
Department=df[['DepartmentDescription']]
N_dpt = len(set(df.DepartmentDescription)) #number of unique departments
Department=np.asarray(Department)

print('Number of unique trip types: %d' %N_y)
print('Number of visits: %d' %N_visit)
print('Number of departments: %d' %N_dpt)

#%% organize samples by VisitNumber, extract features
#for each visit, get:
#ID: VisitNumber
#X_day: Mon-Sun
#N_item:number of items purchased
#N_dept: number of department visited
#Return: if anything was returned
#Multi: if multiple quantities were purchased for one item
#Each_dept: if each department is visited (based on unique(Department)) (69)

N=N_visit
ID=np.zeros(N)
y=np.zeros(N)
X_day=np.zeros(N)
N_item=np.zeros(N)
N_dept=np.zeros(N)
Return=np.zeros(N)
Multi=np.zeros(N)
Tarshop=np.zeros(N)
Each_dept=np.zeros((N,N_dpt))

Days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
Dept=set(df.DepartmentDescription)
Dept=list(Dept)  #each unique departments

t=0
for i in np.unique(Visit):
    print(i)
    ID[t]=i #VisitNumber
    idx=np.where(df.VisitNumber==i)[0] #index for this visit
    y[t]=df.TripType[idx[0]]
    Wd=df.Weekday[idx[0]]
    X_day[t]=Days.index(Wd)
    
    N_item[t]=len(idx)#sum(df.ScanCount[idx].values)
    
    N_dept[t]=len(set(df.DepartmentDescription[idx]))
    Return[t]=np.any(ScanCount[idx]==-1)
    
    if any(df.ScanCount[idx]>1):
        Multi[t]=1
 
    v=set(df.DepartmentDescription[idx]) #unique department visited

    kk=[]
    for k in v:
        u=len(np.where(Department[idx]==k)[0])
        kk.append(u)
        
    if sum(kk)!=0:  
        if max(kk)/sum(kk)>.5:
            Tarshop[t]=1
    else:
        Tarshop[t]=1
    
    for j in np.arange(N_dpt):
        if np.any(df.DepartmentDescription[idx]==Dept[j]):            
           Each_dept[t,j]=1 
           
    t=t+1
#%% trip type distribution
N_type=np.zeros(N_y)
which_type = []
for i in range(N_y): #counts of each trip type
    N_type[i]=np.where(y==np.unique(y)[i])[0].shape[0]
    which_type.append(np.unique(y)[i])
#   
plt.bar(np.arange(N_y),N_type)
plt.title('Trip type distribution')
plt.show()

#%% prepare data for classification

#convert X_day to 0/1 for the 7 days (did not change pred acc)
Weekday=np.zeros((N,7))
for i in np.arange(N):
    Weekday[i,X_day[i]]=1
    
#X_all=np.vstack((Weekday.T,N_item,Return,N_dept,Tarshop,Multi,Each_dept.T)).T # samples x features
X_all=np.vstack((X_day,N_item,Return,N_dept,Tarshop,Multi,Each_dept.T)).T # samples x features
#X_all=np.vstack((X_day,N_item,Return,N_dept,Each_dept.T)).T # samples x features
#X_all=np.vstack((X_day,N_item,Return,N_dept,Tarshop,Multi)).T # samples x features
col=['Weekday','N_item','Return','N_dept','Tarshop','Multi','Department']

#%% transform triptype to 0-37
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y = le.fit_transform(y) #transform original target to 0-37

#split training data into training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X_all,y,test_size=.3)

#feature scaling (standardize for logistic regression)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

#%% fit a logistic regression model (acc = .61, Mis: 11130)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C= 10.0)
lr.fit(X_train_std,y_train)

#predict classes using the training data
y_pred_lr=lr.predict(X_test_std)
print('Misclassfied samples :%d' %(y_test!=y_pred_lr).sum())
from sklearn.metrics import accuracy_score
print('Training Accuracy: %.2f' %lr.score(X_train_std,y_train))
print('Testing Accuracy: %.2f' %accuracy_score(y_test,y_pred_lr))
 
#plot accuracy for each type
lr_acc_type=np.zeros(N_y)
for i in np.arange(N_y):
    which_type =np.unique(y_test)[i]
    idx=np.where(y_test==which_type)[0] #find trials with this type
    lr_acc_type[i]=sum(y_pred_lr[idx]==y_test[idx])/len(idx) 
    
plt.bar(np.arange(N_y),lr_acc_type)
plt.title('Logistic Regression Accuracy')
plt.xlabel('type')
plt.show()

prob_pred_train=lr.predict_proba(X_train_std)
prob_pred_test=lr.predict_proba(X_test_std)


#%% plot prob_train and acc based on threshold
#thre=np.arange(0,1,.1)
#acc=[]
#for i in thre:
#    id=np.where((prob_pred_train>=i) & (prob_pred_train<i+.1))[0]

#%%   
bp()

#%% fit a random forest model (acc = .62, Mis: 10888)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion = 'entropy',n_estimators = 1000,random_state=1,n_jobs = 2)
forest.fit(X_train,y_train)
y_pred_rf = forest.predict(X_test)

print('Misclassfied samples :%d' %(y_test!=y_pred_rf).sum())
from sklearn.metrics import accuracy_score
print('Training Accuracy: %.2f' %forest.score(X_train,y_train))
print('Testing Accuracy: %.2f' %accuracy_score(y_test,y_pred_rf))


importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(len(col)):
    print("%2d) %-*s %f" % (f+1, 30, col[f],importances[indices[f]]))
    
#plot accuracy for each type
rf_acc_type=np.zeros(N_y)
for i in np.arange(N_y):
    which_type =np.unique(y_test)[i]
    idx=np.where(y_test==which_type)[0] #find trials with this type
    rf_acc_type[i]=sum(y_pred_rf[idx]==y_test[idx])/len(idx) 
    
plt.bar(np.arange(N_y),rf_acc_type)
plt.title('Random Forest Accuracy')
plt.xlabel('type')
plt.show()

#plt.scatter(lr_acc_type,rf_acc_type)
#plt.plot(np.arange(0,1,.1),np.arange(0,1,.1))
#plt.xlabel('LR acc')
#plt.ylabel('RF acc')
#plt.show()


#%%: fit SVM with gaussian kernel 
from sklearn.svm import SVC
svm = SVC(kernel='rbf',C=1.0,random_state=0)
svm.fit(X_train_std,y_train)

y_pred=svm.predict(X_test_std)
print('Misclassfied SVM samples :%d' %(y_test!=y_pred).sum())
print('Tesing Accuracy: %.2f' %accuracy_score(y_test,y_pred))


#%% Bagging (test: .604)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion = 'entropy',max_depth =None)
bag = BaggingClassifier(base_estimator = tree,n_estimators=1000,
                      max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                          bootstrap_features=False,
                       n_jobs=1,
                       random_state=1)
from sklearn.metrics import accuracy_score
tree = tree.fit(X_train,y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train,y_train_pred)
tree_test = accuracy_score(y_test,y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))                       
                       
bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test))

#%% fit a KNN model (acc = .53, Mis: 13409)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5,p = 2,metric='minkowski')
knn.fit(X_train_std,y_train)
y_pred_knn = knn.predict(X_test_std)

print('Misclassfied samples :%d' %(y_test!=y_pred_knn).sum())
from sklearn.metrics import accuracy_score
print('Training Accuracy: %.2f' %knn.score(X_train_std,y_train))
print('Testing Accuracy: %.2f' %accuracy_score(y_test,y_pred_knn))

#plot accuracy for each type
knn_acc_type=np.zeros(N_y)
for i in np.arange(N_y):
    which_type =np.unique(y_test)[i]
    idx=np.where(y_test==which_type)[0] #find trials with this type
    knn_acc_type[i]=sum(y_pred_knn[idx]==y_test[idx])/len(idx) 
    
plt.bar(np.arange(N_y),knn_acc_type)
plt.title('KNN Accuracy')
plt.xlabel('type')
plt.show()

plt.scatter(lr_acc_type,knn_acc_type)
plt.plot(np.arange(0,1,.1),np.arange(0,1,.1))
plt.xlabel('LR acc')
plt.ylabel('KNN acc')
plt.show()

#%%: visualize for each trip type: weekday, item puchased, return or not
Trip_day=np.zeros((N_y,7))
for i in np.arange(38): #loop through each trip type
    id=np.where(y==np.unique(y)[i])[0] #find visits with this trip type
    for _ in np.arange(7):
      Trip_day[i][_]=int(np.where(X_day[id]==_)[0].shape[0])

#%%
for i in np.arange(38):    
    plt.bar(np.arange(7),Trip_day[i])
    plt.title('Trip type %d' %np.unique(y)[i])
    plt.xticks(np.arange(7),Days)
    plt.show() 
    
