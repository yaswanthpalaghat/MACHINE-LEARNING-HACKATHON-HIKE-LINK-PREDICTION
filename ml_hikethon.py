#Author: Yaswanth Sai Palaghat

import zipfile
import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
 
zf = zipfile.ZipFile('train.zip') 
train_df = pd.read_csv(zf.open('train.csv'))
train_features = pd.read_csv(zf.open('user_features.csv'))



t2 = train_df.iloc[:100000,:]
train_features = train_features.set_index('node_id')

cols = train_features.columns
for col in cols:
  t2['1'+col] = t2['node1_id'].apply(lambda x: train_features.loc[x,col])

for col in cols:
  t2['2'+col] = t2['node2_id'].apply(lambda x: train_features.loc[x,col])

t2.head()



X = t2.iloc[:,3:].values.astype(float)
Y = t2.loc[:,'is_chat'].values

X.shape
seed = 7

model = Sequential()
model.add(Dense(60, input_dim=26, kernel_initializer='normal', activation='relu'))
model.add(Dense(100, input_dim=60, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, input_dim=100, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X,Y, epochs = 20, batch_size = 1000)

model.predict(X)


 
zf2 = zipfile.ZipFile('test.zip') 
test_df = pd.read_csv(zf2.open('test.csv'))
# train_features = pd.read_csv(zf.open('user_features.csv'))

test_df.shape

l =[]
t =0
for i in range(59):
  print(i)
  test1 = test_df.iloc[t:t+200000,:]


  for col in cols:
    test1['1'+col] = test1['node1_id'].apply(lambda x: train_features.loc[x,col])

  for col in cols:
    test1['2'+col] = test1['node2_id'].apply(lambda x: train_features.loc[x,col])

  l.append(model.predict(test1.iloc[:,3:].values))
  t+=200000

len(l)

59*200000

test_df.shape

flat_list = [item for sublist in l for item in sublist]

len(flat_list)

test_df['Score'] = flat_list
test_df.head()

test_f = test_df[['id','Score']]
test_f.shape

# test_f['Score'] = test_f['Score'].values
type(test_f.Score[0])

test_f['is_chat'] = test_f['Score'].apply(lambda x: x[0])

test_f.drop(columns = 'Score', inplace = True)
test_f.head()

test_f[['id','is_chat']].to_csv('prediction.csv', index= False)



