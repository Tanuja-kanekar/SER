#!/usr/bin/env python
# coding: utf-8

# ## Speech Emotion Recognition

# ### Importing the libraries

# In[1]:


#loading the necessary libraries
import librosa 
from librosa import display
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import sys
import IPython.display as ipd


# In[2]:


#to get the path of current working directory
path = os.getcwd()
print(path)


# In[3]:


#saving the location of our data folder
Data1="/home/thanuja/Downloads/output/train/"
Data2="/home/thanuja/Downloads/output/test/"
Data3="/home/thanuja/Downloads/output/val/"


# In[4]:



dir_train=os.listdir(Data1)
dir_test=os.listdir(Data2)
dir_val=os.listdir(Data3)

print(dir_val)


# ### Train Data 

# In[5]:



emotion=[]
path=[]

for sub_dir1 in dir_train:
    filename=os.listdir(Data1+sub_dir1)
    for files in filename:
        part=files.split('.')[0].split('-')
        emotion.append(int(part[2]))
        path.append(Data1+sub_dir1+'/'+files)
                
df=pd.DataFrame(emotion,columns=['Emotions'])
pathdf=pd.DataFrame(path,columns=['Path'])
train_data=pd.concat([pathdf,df],axis=1)


# In[6]:


train_data.head()


# In[7]:


train_data.shape


# In[8]:


train_data.nunique()


# In[9]:


train_data=train_data.replace({2:'calm',3:'happy',4:'sad',5:'angry'
                ,7:'disgust',8:'surprise'})


# In[10]:


train_data


# In[11]:


train_data['Emotions'].value_counts()


# ### Test Data

# In[12]:


emotion1=[]
path1=[]

for sub_dir2 in dir_test:
    filename=os.listdir(Data2+sub_dir2)
    for files in filename:
        part=files.split('.')[0].split('-')
        emotion1.append(int(part[2]))
        path1.append(Data2+sub_dir2+'/'+files)
                


# In[13]:


df1=pd.DataFrame(emotion1,columns=['Emotions'])
pathdf1=pd.DataFrame(path1,columns=['Path'])
test_data=pd.concat([pathdf1,df1],axis=1)


# In[14]:


test_data.head()


# In[15]:


test_data=test_data.replace({2:'calm',3:'happy',4:'sad',5:'angry'
                ,7:'disgust',8:'surprise'})


# In[16]:


test_data.nunique()


# ### Validation Data

# In[17]:


emotion2=[]
path2=[]

for sub_dir3 in dir_val:
    filename=os.listdir(Data3+sub_dir3)
    for files in filename:
        part=files.split('.')[0].split('-')
        emotion2.append(int(part[2]))
        path2.append(Data3+sub_dir3+'/'+files)
                
df2=pd.DataFrame(emotion2,columns=['Emotions'])
pathdf2=pd.DataFrame(path2,columns=['Path'])
valid_data=pd.concat([pathdf2,df2],axis=1)


# In[18]:


valid_data.head()


# In[19]:


valid_data=valid_data.replace({2:'calm',3:'happy',4:'sad',5:'angry'
                ,7:'disgust',8:'surprise'})


# In[20]:


Data_valid = valid_data.reindex(np.random.permutation(valid_data.index))
Data_valid


# In[21]:


Data_valid.to_csv("Data3.csv",index=False)


# In[22]:


speech_data_valid=pd.read_csv("Data3.csv")
speech_data_valid


# ### Train and Test data

# In[23]:


Data=pd.concat([train_data,test_data],ignore_index=True)
Data[120:125]


# ### Shuffling the data

# In[24]:


Data = Data.reindex(np.random.permutation(Data.index))
Data


# ### Saving the shuffled data

# In[25]:


Data.to_csv("Data2.csv",index=False)


# In[26]:


speech_data=pd.read_csv("Data2.csv")
speech_data


# ### Feature Extraction

# In[27]:


Feature_data = pd.DataFrame(columns=['Features'])

counter = 0
for index, path in enumerate(speech_data.Path):
    X, sample_rate = librosa.load(path,res_type='kaiser_fast')
    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=40).T, axis=0)
    Feature_data.loc[counter]=[mfccs]
    counter=counter+1


# In[28]:


Feature_data.head()


# In[29]:


Feature_data=pd.concat([pd.DataFrame(Feature_data['Features'].values.tolist()),speech_data.Emotions],axis=1)
Feature_data


# ### Input data

# In[30]:


X_data=Feature_data.drop(['Emotions'],axis=1)
X_data


# In[31]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scale=scaler.fit_transform(X_data)


# In[32]:


X_data1= pd.DataFrame(scale)
X_data1


# ### Target Data

# In[33]:


Y_data=Feature_data.Emotions
Y_data.head()


# ### Train and Test split

# In[34]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_data1,Y_data,test_size=0.2,random_state=40)


# In[35]:


print((x_train.shape,x_test.shape,y_train.shape,y_test.shape))


# ## Model Building

# #### Support Vector Machine for non standardized data

# #### SVC rbf

# In[36]:


from sklearn.svm import SVC
svc_model=SVC(C=10,gamma=0.001,kernel='rbf').fit(x_train,y_train)

print('accuracy: {}'.format(svc_model.score(x_test,y_test)))


# In[37]:


train_acc = float(svc_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# #### SVC Linear

# In[38]:


from sklearn.svm import SVC
svc_model=SVC(C=1,gamma=0.0001,kernel='linear').fit(x_train,y_train)

print('accuracy: {}'.format(svc_model.score(x_test,y_test)))


# In[39]:


train_acc = float(svc_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# #### Random Forest Classifier

# In[40]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators= 200,max_depth=4).fit(x_train,y_train)


# In[41]:


print('accuracy: {}'.format(model.score(x_test,y_test)))


# In[42]:


train_acc = float(model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# ### XGBoost Classifier

# In[43]:


from xgboost import XGBClassifier


# In[44]:


xg_model = XGBClassifier(learning_rate=3.5,n_estimators=300,max_depth=4).fit(x_train, y_train)


# In[45]:


print('accuracy: {}'.format(xg_model.score(x_test,y_test)))


# In[46]:


train_acc = float(xg_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(xg_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# #### Prediction on Validation data

# In[47]:


final_model=SVC(C=10,gamma=0.001,kernel='rbf').fit(X_data1,Y_data)


# In[48]:


import pickle

pickle.dump(final_model,open('final_ser.pkl','wb'))


# In[49]:


modelser=pickle.load(open('final_ser.pkl','rb'))


# In[50]:


Feature_data_valid = pd.DataFrame(columns=['Features_valid'])

counter = 0
for index, path in enumerate(speech_data_valid.Path):
    X, sample_rate = librosa.load(path,res_type='kaiser_fast')
    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=40).T, axis=0)
    Feature_data_valid.loc[counter]=[mfccs]
    counter=counter+1


# In[69]:


Feature_data1=pd.DataFrame(Feature_data_valid['Features_valid'].values.tolist())
Feature_data1.head()


# In[58]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scale=scaler.fit_transform(Feature_data1)


# In[59]:


SER=modelser.predict(scale)


# In[68]:


SER1=pd.DataFrame(SER,columns=['emo'])
SER1.head()


# In[67]:


Actual_valid=pd.DataFrame(speech_data_valid.Emotions)
Actual_valid.head()


# #### Actual Validation emotions versus predicted emotions

# In[66]:


final_df=Actual_valid.join(SER1)
final_df


# In[ ]:




