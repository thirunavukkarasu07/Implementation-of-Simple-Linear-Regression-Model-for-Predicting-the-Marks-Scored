# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')

#displaying the content in datafile
df.head()
df.tail()

#segregating data to variables
X = df.iloc[:,:-1].values
print(X)

Y=df.iloc[:,1].values
print(Y)

#splitting train and test data

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
print(Y_pred)

#display actual values
print(Y_test)

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

#Graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(X_test,Y_test,color="pink")
plt.plot(X_test,Y_pred,color="black")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Thirunavukkarasu Meenakshisundaram
RegisterNumber: 212224220117
*/
```

## Output:
HEAD

![image](https://github.com/user-attachments/assets/42ac0396-8a3c-486c-9f90-cc4314f4a960)

TAIL

![image](https://github.com/user-attachments/assets/23100b22-225a-414d-b6d3-3ffd70207cf9)

Segregating data to variables

![image](https://github.com/user-attachments/assets/1f1878b7-c0b1-46f2-8d18-8bf8f4ae4c50)

Displaying predicted values

![image](https://github.com/user-attachments/assets/ed3e4e8d-515a-49fa-8a5e-6e8dfe4208c9)

displaying actual values

![image](https://github.com/user-attachments/assets/8b004ad5-1208-408f-8e89-7d3435fb5371)

MSE MAE RMSE

![image](https://github.com/user-attachments/assets/d9e838a0-749a-4204-b64f-ce8c38d84854)

Graph plot for training data

![image](https://github.com/user-attachments/assets/3c3d84fb-e4bc-4b65-a434-b7c0f74100c6)

Graph plot for test data

![image](https://github.com/user-attachments/assets/d4b4e9c0-4c88-4e58-97c1-d10a7adf2ebf)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
