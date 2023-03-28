import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# load the dataset
data = pd.read_csv("Churn_Modelling.csv")
# select the independent variable
X=data[['Age']]
# select the dependent variable
y=data['EstimatedSalary']
#split the data into training and test data set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#created a linear regression model
model= LinearRegression()
#train the model using the training data
model.fit(X_train,y_train)
#predict the target on the test data set
y_pred = model.predict(X_test)
print("Age:", X_test.values[0])
print("Actual Salary:",y_test.values[0])
print("Predicted Salary:", round(y_pred[0],2))
print("Accuracy:", model.score(X_test, y_test))