import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print("Fazlullah Bokhari")

credit_data = pd.read_csv('credit_data.csv')

# to show top 3 rows of data
print(credit_data.head(3))
# Describing the data
print(credit_data.describe())
# printing the correlation matrices of data
print(credit_data.corr())


features = credit_data[['income','age','loan']]
print("Features: ",features)
target = credit_data.default
print("Target: ",target)
# 30% of data for testing 70% of data for training
features_train, features_test, target_train,target_test = train_test_split(features,target,test_size=0.3,random_state=0)

model = LogisticRegression()
model.fit = model.fit(features_train,target_train)

print("b0 is: ",model.intercept_)
print("b1 is: ",model.coef_)


predictions = model.fit.predict(features_test)

print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))
