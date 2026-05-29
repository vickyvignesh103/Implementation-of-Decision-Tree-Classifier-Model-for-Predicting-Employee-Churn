# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the employee dataset using dummy variable encoding.

2.Split the dataset into training and testing data and initialize the Decision Tree classifier.

3.Train the Decision Tree model using the training dataset.

4.Predict test results, calculate accuracy, and visualize the Decision Tree.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VIGNESH V
RegisterNumber:  212225230298
*/
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()



print("data.info():")
data.info()



print("isnull() and sum():")
data.isnull().sum()



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()



print("x.head():")
# Fixed: Added closing quote to complete the string literal "Work_accident"
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident"]]
x.head()



y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy



print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0]])  # Removed the last 2 values (1,2)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```

## Output:
<img width="881" height="291" alt="image" src="https://github.com/user-attachments/assets/091b7367-c0e8-49a2-a891-96f8ec3fa1ea" />
<img width="480" height="393" alt="image" src="https://github.com/user-attachments/assets/f84959de-860c-479a-a261-9cbd4a2eef7f" />
<img width="282" height="282" alt="image" src="https://github.com/user-attachments/assets/c397e079-8146-469f-8823-98bdaba6c422" />
<img width="258" height="122" alt="image" src="https://github.com/user-attachments/assets/d8e0033b-6e8c-44cc-b22f-315e4793034a" />
<img width="887" height="260" alt="image" src="https://github.com/user-attachments/assets/37efe424-e20a-4940-9982-65783bea6c29" />
<img width="908" height="257" alt="image" src="https://github.com/user-attachments/assets/17b330a5-61e0-4bc8-a898-e86bbb1b1482" />
<img width="197" height="67" alt="image" src="https://github.com/user-attachments/assets/a9f13bd5-87f8-4cdf-8916-208f2b77e09b" />
<img width="907" height="682" alt="image" src="https://github.com/user-attachments/assets/151cfc10-2aeb-4c7d-b453-549c2ddc802a" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
