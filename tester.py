#Import required packages
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
boston_dataset = load_boston()

# Getting the data ready
# Generate train dummy data for 1000 Students and dummy test for 500
# #Columns :Age, Hours of Study &Avg Previous test scores
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

boston['MEDV']=boston_dataset.target

train_y = boston.pop('MEDV')
train_x=boston
data=np.loadtxt("/Users/omarahmad/Downloads/machine-learning-ex1/ex1/ex1data1.txt",dtype="i",delimiter=",")


np.random.seed(2018)
#Setting seed for reproducibility
#train_data, test_data = np.random.random((1000, 3)), np.random.random((500, 3))
train_data=data[:,0]

a=np.mean(train_data,axis=0)
b=np.std(train_data,axis=0)
train_data=(train_data-a)/b

#Generate dummy results for 1000 students : Whether Passed (1) or Failed (0)
#labels = np.random.randint(2, size=(1000, 1))
labels= data[:,1]

#Defining the model structure with the required layers, # of neurons, activation function and optimizers
model = Sequential()
model.add(Dense(5, input_dim=13, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#Train the model and make predictions
model.fit(train_x, train_y, epochs=10, batch_size=32)
#Make predictions from the trained
print("test data predictions")
#modelpredictions = model.predict(test_data)

