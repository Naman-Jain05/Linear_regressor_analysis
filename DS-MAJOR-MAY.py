# creatig dataframe
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/Mall_Customers.csv')
print(df)

#data visualization
import matplotlib.pyplot as plt
plt.figure(1)
plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'])
plt.title('Annual Income (k$) vs Spending Score')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")

#input x output y
x=df.iloc[:,3:4].values
y=df.iloc[:,4].values

#train and test variables
from sklearn.model_selection import train_test_split
x_train ,x_test ,y_train ,y_test =train_test_split(x,y,random_state=0)

#scaling input data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#running regressor (linear regressor)
from sklearn.linear_model import LinearRegression
model=LinearRegression()

#fit the model
model.fit(x_train,y_train) 

#predict output
y_pred = model.predict(x_test)
print("\nactual data:")
print(y_test)
print("\ndata predicted by linear regressor:")
print(y_pred)

#y=mx+c  line
#y - dependent variable
#m - slope
#c - y-intercept
#x - independent variable

#to find out m 
print("\nm:",model.coef_)

#y -intercept (C)
print("\nc:",model.intercept_)

#evaluation : accuracy score
plt.figure(2)
plt.title('Linear regressor line')
plt.xlabel("Annual Income (k$) scaled")
plt.ylabel("Spending Score")
plt.scatter(x_train,y_train,c='cyan')
plt.plot(x_test,y_pred,c='lime')
plt.show()

#conclusion:
#            linear regressor for given data is: y=-4.62x+51.99