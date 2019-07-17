# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:28:21 2019

@author: Heisenberg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
test=pd.read_excel('Data_Test2.xlsx')

train=pd.read_excel('Data_Train2.xlsx')
X=train.iloc[:,2]
y=train.iloc[:,7]

len(train['TITLE_y'].unique())
len(train['CUISINES_y'].unique())
len(train['CITY'].unique())
len(train['LOCALITY'].unique())
len(train['RATING'].unique())
train['Seats'].unique()

len(test['TITLE_y'].unique())
len(test['CUISINES_y'].unique())
len(test['CITY'].unique())
test['Transmission'].unique()
test['RATING'].unique()
test['Seats'].unique()


del train['CITY']
del train['LOCALITY']
del test['CITY']
del test['LOCALITY']


#train['CUISINES_y'] = pd.factorize(train['CUISINES_y'])[0] 

#train['CUISINES']=train['CUISINES'].str.split(",", expand = True)
#train['CUISINES']=train['CUISINES'].apply(lambda x: train['CUISINES'])
#train.set_index("Restaurant_ID")
#new_df = pd.DataFrame(train.CUISINES.str.split(',').tolist(),index=train.RESTAURANT_ID).stack()
#new_df = new_df.reset_index([0, 'RESTAURANT_ID'])
#new_df.columns = ['RESTAURANT_ID', 'CUISINES']

#train = pd.merge(train,new_df, how='inner', on='RESTAURANT_ID')
#del train['CUISINES_x']

new_df2 = pd.DataFrame(train.TITLE.str.split(',').tolist(),index=train.RESTAURANT_ID).stack()
new_df2 = new_df2.reset_index([0, 'RESTAURANT_ID'])
new_df2.columns = ['RESTAURANT_ID', 'TITLE']

train = pd.merge(train,new_df2, how='inner', on='RESTAURANT_ID')
del train['TITLE_x']

train=train.drop_duplicates()


#test_new_df = pd.DataFrame(test.CUISINES.str.split(',').tolist(),index=test.RESTAURANT_ID).stack()
#test_new_df = test_new_df.reset_index([0, 'RESTAURANT_ID'])
#test_new_df.columns = ['RESTAURANT_ID', 'CUISINES']

#test = pd.merge(test,test_new_df, how='inner', on='RESTAURANT_ID')
#del test['CUISINES_x']

test_new_df2 = pd.DataFrame(test.TITLE.str.split(',').tolist(),index=test.RESTAURANT_ID).stack()
test_new_df2 = test_new_df2.reset_index([0, 'RESTAURANT_ID'])
test_new_df2.columns = ['RESTAURANT_ID', 'TITLE']

test = pd.merge(test,test_new_df2, how='inner', on='RESTAURANT_ID')
del test['TITLE_x']

test=test.drop_duplicates()


train['VOTES']=train['VOTES'].str.extract('(\d*.\d+)')
train['VOTES'] = pd.to_numeric(train['VOTES'], downcast='signed',errors='coerce')



test['VOTES']=test['VOTES'].str.extract('(\d*.\d+)')
test['VOTES'] = pd.to_numeric(test['VOTES'], downcast='signed',errors='coerce')
train['RATING'].fillna('3.9',inplace=True)
train['VOTES'].fillna(int(np.mean(train['VOTES'])),inplace=True)
train['RATING'].replace('NEW','3.9', inplace = True)
X=train[train['RATING'] == "-"]

train.drop(X.index, axis=0 ,inplace=True)

test['RATING'].fillna('4.0',inplace=True)
test['VOTES'].fillna(int(np.mean(test['VOTES'])),inplace=True)
test['RATING'].replace('NEW','4.0', inplace = True)
y=test[test['RATING'] == "-"]

test.drop(y.index, axis=0 ,inplace=True)

test.apply(lambda x: sum(x.isnull()))
train.apply(lambda x: sum(x.isnull()))



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
#train['CUISINES_y'] = labelencoder_x.fit_transform(train['CUISINES_y'])
train['TITLE_y'] = labelencoder_x.fit_transform(train['TITLE_y'])

#dummy1=pd.get_dummies(train['CUISINES_y'],prefix='CUISINES')
dummy2=pd.get_dummies(train['TITLE_y'],prefix='TITLE')


#train = pd.concat([train,dummy1],axis=1)
train = pd.concat([train,dummy2],axis=1)

#test['CUISINES_y'] = labelencoder_x.fit_transform(test['CUISINES_y'])
test['TITLE_y'] = labelencoder_x.fit_transform(test['TITLE_y'])

#test_dummy1=pd.get_dummies(test['CUISINES_y'],prefix='CUISINES')
test_dummy2=pd.get_dummies(test['TITLE_y'],prefix='TITLE')


#test = pd.concat([test,test_dummy1],axis=1)
test = pd.concat([test,test_dummy2],axis=1)

#train.where(train["CUISINES_y"]==test["CUISINES_y"])


#del train['CUISINES_y']
del train['TITLE_y']
#del test['CUISINES_y']
del test['TITLE_y']

del train['RESTAURANT_ID']
del test['RESTAURANT_ID']

"""dummy_Cuisines=train.iloc[:,4:223]
dummy_Title=train.iloc[:,224:248]
dummy_Cuisines_test=test.iloc[:,3:191]
dummy_Title_test=test.iloc[:,192:]

dummy_Cuisines[dummy_Cuisines.isin(dummy_Cuisines_test)]
intersection=pd.merge(dummy_Cuisines, dummy_Cuisines_test, how='left')"""
#Scale
#from sklearn.preprocessing import StandardScaler
#sc_X=StandardScaler()
#train.iloc[:,train.columns != 'COST']=sc_X.fit_transform(train.iloc[:,train.columns != 'COST'])
#test=sc_X.transform(test)

#PCA
#from sklearn.decomposition import PCA
#pca = PCA(n_components = None)
#train.iloc[:,train.columns != 'COST'] = pca.fit_transform(train.iloc[:,train.columns != 'COST'])
#test = pca.transform(test)
#explained_variance = pca.explained_variance_ratio_

#Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 512, random_state = 0)
regressor.fit(train.iloc[:,train.columns != 'COST'], train.iloc[:,3])



# Predicting the Test set results
Z=test[test['RATING'] == "-"]

y_pred = regressor.predict(test)
y_pred=pd.DataFrame(y_pred)
submission=y_pred.to_excel("submission.xlsx")

#364
