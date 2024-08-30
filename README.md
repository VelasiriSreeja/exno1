# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
## DATA CLEANING PROCESS
```py
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
```
![1](https://github.com/user-attachments/assets/946e7748-d8df-4b8e-9006-35fdedae92fd)

```py
df.head(5)
```
![2](https://github.com/user-attachments/assets/278edd35-898a-4a3f-b5f6-15dcc8f8a78a)

```py
df.tail(5)
```
![3](https://github.com/user-attachments/assets/9b4a4706-52e6-4009-a7e2-a5b54a919bd6)

```py
df.info()
```
![4](https://github.com/user-attachments/assets/ea439663-0559-418f-be25-fc34ec0c8e97)

```py
df.describe()
```
![5](https://github.com/user-attachments/assets/b103ac7c-d09b-46ee-a50e-c1574e9a245d)

```py
df.shape
```
(21, 12)

```py
df.isnull().sum()
```
![6](https://github.com/user-attachments/assets/3e272276-b05f-46be-b35c-bab5302aebd1)

```py
df.nunique()
```
![7](https://github.com/user-attachments/assets/f2d4da76-b46b-48d1-bba5-caa1ac6de667)

```py
df['GENDER'].value_counts()
```
![8](https://github.com/user-attachments/assets/1767fdb6-19de-4b7b-ac56-9d1975d7ba24)

```py
df.dropna(how='any').shape
```
(13, 12)

```py
df.shape
```
(21, 12)

```py
x1=df.dropna(how='any')
x1
```
![9](https://github.com/user-attachments/assets/c43bff8c-0c25-46ac-886c-f863c189b15b)

```py
x2=df.dropna(how='all')
x2
```
![10](https://github.com/user-attachments/assets/3fd2d03f-bfd7-4c4b-8da0-ccd86479f16c)

```py
tot=df.dropna(subset=['TOTAL'],how='any')
tot
```
![11](https://github.com/user-attachments/assets/48ffee62-c34a-4bb4-a265-03c18d40f866)

```py
tot=df.dropna(subset=['M1','M2','M3','M4'],how='any')
tot
```
![12](https://github.com/user-attachments/assets/4c399958-f109-40e8-bc76-e484af0125cf)

```py
s=df.fillna(method='ffill')
s
```
![13](https://github.com/user-attachments/assets/93cbbbb2-8383-4b1a-9e0a-8b09cf7ef4f5)

```py
df.isna().sum()
```
![14](https://github.com/user-attachments/assets/2dcbbd0a-cdd1-4cff-8b61-e91040faa72c)

```py
df['M1']
```
![15](https://github.com/user-attachments/assets/a548b1bb-18ce-44e5-aa62-9fea78d7f2c4)

```py
df.isnull()
```
![16](https://github.com/user-attachments/assets/0b236a3f-2e84-4906-a497-8d230cf65318)

```py
df.notnull()
```
![17](https://github.com/user-attachments/assets/796be0e8-eed2-49e6-b4e6-1e61d9ec3ce6)

```py
x1=df.dropna(axis=0)
x1
```
![18](https://github.com/user-attachments/assets/9d9cfcbf-d79a-4828-a3e2-23a7b65f8c35)

```py
df.duplicated()
```
![19](https://github.com/user-attachments/assets/745d2096-8fb5-4030-8021-25bfa963e68a)

```py
m=df.drop_duplicates(inplace=False)
m
```
![20](https://github.com/user-attachments/assets/6bc4230b-d05f-49de-9ecb-83ba0e4b6e1d)

```py
import seaborn as sns
sns.heatmap(df.isnull(),yticklabels=False,annot=True)
```
![image](https://github.com/user-attachments/assets/a12b151a-9edc-4281-92b2-543d7bb21fa4)

```py
df.dropna(inplace=True)
```
```py
sns.heatmap(df.isnull(), yticklabels=False, annot=True)
```
![image](https://github.com/user-attachments/assets/d8b6d2f3-9dac-4148-947a-3cdb68a20297)

```py
df.loc[0:3]
```
![21](https://github.com/user-attachments/assets/8be1a9ce-2df2-4169-b819-f7a192672d53)

```py
df.dtypes
```
![22](https://github.com/user-attachments/assets/85786c10-2262-4834-ae20-685535e92ddf)

```py
df.filter(regex='a', axis=1)
```
![23](https://github.com/user-attachments/assets/d441788b-43a6-44b9-8e0e-147108704d65)

## OUTLIERS DETECTION AND REMOVAL
### IQR METHOD
```py
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
ir=pd.read_csv("iris.csv")
ir
```
![1](https://github.com/user-attachments/assets/abbe1ea4-47a3-410d-b2db-ad3ac2147be3)

```py
sns.boxplot(x='sepal_width',data=ir)
```
![image](https://github.com/user-attachments/assets/ae21615c-5f44-4075-94f2-d60209ac56eb)

```py
sns.boxenplot(x='sepal_width',data=ir)
```
![image](https://github.com/user-attachments/assets/fc17fd9d-cd69-4955-9ca6-4c176765c5b7)

```py
Q1=np.percentile(ir.sepal_width,25)
Q2=np.percentile(ir.sepal_width,50)
Q3=np.percentile(ir.sepal_width,75)
IQR=Q3-Q1
IQR
```
0.5

```py
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```
```py
outliers = [x for x in ir.sepal_width if x < lower_bound or x > upper_bound]
```

```py
print("Q1:",Q1)
print("Q2:",Q2)
print("Q3:",Q3)
print("IQR:",IQR)
print("Lower Bound:",lower_bound)
print("Upper Bound:",upper_bound)
print("Outliers:",outliers)
```
Q1: 2.8

Q2: 3.0

Q3: 3.3

IQR: 0.5

Lower Bound: 2.05

Upper Bound: 4.05

Outliers: [4.4, 4.1, 4.2, 2.0]

```py
ir=ir[((ir.sepal_width>=lower_bound)&(ir.sepal_width<=upper_bound))]
ir
```
![2](https://github.com/user-attachments/assets/2c9bb806-f599-437c-8125-08e84cb4c696)

```py
ir.dropna()
```
![3](https://github.com/user-attachments/assets/dfc914be-d21f-4068-9934-eae4552d31ca)

```py
sns.boxplot(x='sepal_width',data=ir)
```
![image](https://github.com/user-attachments/assets/94449ef2-f77a-4239-a0eb-ca974ae16ff2)

```py
sns.boxenplot(x='sepal_width',data=ir)
```
![image](https://github.com/user-attachments/assets/e717ce75-f91b-4330-883c-5d9c19b4f249)

# Result
   Thus the program to perform data cleaning and save the cleaned data to a file is successfully executed.
