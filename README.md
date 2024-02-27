<H3>NAME     :  DHANUSH P</H3>
<H3>REG NO   :  212222040034</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling (1).csv", index_col="RowNumber")
df
#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
#Checking for null values
df.isnull().sum()
#Checking for duplicate values
df.duplicated()
#Describing the dataset
df.describe()
#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```


## OUTPUT:
### DataSet
![image](https://github.com/DhanushPalani/Ex-1-NN/assets/121594640/13c95d9f-177c-49ff-b912-0e6c7e85b5a3)
### Dropping Unwanted Features
![image](https://github.com/DhanushPalani/Ex-1-NN/assets/121594640/24dd557d-d932-4c08-b4ae-9754cc7780c7)
### Checking for Null Values
![image](https://github.com/DhanushPalani/Ex-1-NN/assets/121594640/321abb7e-d745-4297-9c33-f5a7ec4fd57e)
### Checking for Duplication
![image](https://github.com/DhanushPalani/Ex-1-NN/assets/121594640/06867b64-fb4f-4860-bfb4-0013ae89ae21)
### Describing the Dataset
![image](https://github.com/DhanushPalani/Ex-1-NN/assets/121594640/362831d4-5904-42c3-8c43-7f82e3990a65)
### Scaling the Values
![image](https://github.com/DhanushPalani/Ex-1-NN/assets/121594640/37f3ff52-4847-4e6b-9bac-bc7b3f8d68c8)
### X Features
![image](https://github.com/DhanushPalani/Ex-1-NN/assets/121594640/4924b898-e206-4d3a-83f5-5b596670abaa)
### Y Features
![image](https://github.com/DhanushPalani/Ex-1-NN/assets/121594640/2b2d9737-d0de-423f-9be1-6e931e7a4c3c)
### Splitting the training and testing dataset
![image](https://github.com/DhanushPalani/Ex-1-NN/assets/121594640/8d50e2bc-26ce-45d9-9a54-3a65c55f0615)











## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


