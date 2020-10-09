#importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#importing the dataset
dataset = pd.read_csv('hiring.csv')

#inplace missing value by 0
dataset['experience'].fillna('zero', inplace = True)

#mean of missing value
dataset['test_score'].fillna(dataset['test_score'].mean(),inplace = True)

# independent variable
X = dataset.iloc[:,:-1]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

# fit in experience column
X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))


y = dataset.iloc[:,-1]

#spliting training set and test set
#since we have too small data so we will train our model with all the availabel data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fitting model with training data
regressor.fit(X, y)

# saviing model to disk
pickle. dump(regressor, open('model.pkl','wb'))

#loading model to prepare the result
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))