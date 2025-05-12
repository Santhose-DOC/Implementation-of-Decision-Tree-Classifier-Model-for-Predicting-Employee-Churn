# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```python
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])

```

Developed by: Santhose Arockiaraj J

RegisterNumber:  212224230248


## Output:
### Data Head :

![Screenshot 2025-05-12 104304](https://github.com/user-attachments/assets/f2ba79f9-a2b0-41a8-b7a8-8a4bd666573d)

### DataSet Info :

![Screenshot 2025-05-12 104334](https://github.com/user-attachments/assets/5959a8ea-8550-4d66-b0bd-9b6d38677a42)

### Null Dataset :

![Screenshot 2025-05-12 104422](https://github.com/user-attachments/assets/5ba6c217-7615-43a2-a8c1-3596c9c097b2)

### Values count in left column :

![Screenshot 2025-05-12 104503](https://github.com/user-attachments/assets/6f080a86-75b0-4515-89ed-7e50c9c3a704)

### Dataset transformed head :

![Screenshot 2025-05-12 104649](https://github.com/user-attachments/assets/9494d118-9b6e-4fe5-9ecb-30370baf5f24)

### x.head :

![Screenshot 2025-05-12 104719](https://github.com/user-attachments/assets/51c7cdc2-4ebd-4a88-a37b-a5e2a9399372)

### Accuracy :

![Screenshot 2025-05-12 104908](https://github.com/user-attachments/assets/35f4588c-7360-4af2-96c8-04ff995a6715)

### Data prediction :

![Screenshot 2025-05-12 104933](https://github.com/user-attachments/assets/bf676a1d-26ec-40d7-9f38-6bd4c1c50521)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
