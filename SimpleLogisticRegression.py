import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
print("Fazlullah Bokhari")


x1 = np.array([0,0.6,1.1,1.5,1.8,2.5,3,3.1,3.9,4,4.9,5,5.1])
print("X1 length: ",len(x1))
y1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
print("Y1 length: ",len(y1))

x2 = np.array([3,3.8,4.4,5.2,5.5,6.5,6,6.1,6.9,7,7.9,8,8.1])
print("X2 length: ",len(x2))
y2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])
print("Y2 length: ",len(y2))

x = np.array([
    [0],[0.6],[1.1],[1.5],[1.8],[2.5],[3],[3.1],[3.9],[4],[4.9],[5],[5.1],
    [3],[3.8],[4.4],[5.2],[5.5],[6.5],[6],[6.1],[6.9],[7],[7.9],[8],[8.1]
    ])
print("X length: ",len(x))
y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1])
print("Y length: ",len(y))

plt.plot(x1,y1,marker="o",color='blue')
plt.plot(x2,y2,marker="o",color='red')
#plt.show()

model = LogisticRegression()
model.fit(x,y)
print("b0 is: ",model.intercept_)
print("b1 is: ",model.coef_)

def logistic(classifier,x):
    return  1/(1+np.exp(-(model.intercept_ + model.coef_ * x)))

for i in range(1,120):
    plt.plot(i/10.0 - 2, logistic(model, i/10.0), marker="o", color='green')

plt.axis([-2,10,-0.5,2])
plt.show()

pred = model.predict([[1]])
print("Prediction: ",pred)
prob_pred = model.predict_proba([[1]])
print("Probability of prediction: ",prob_pred)
