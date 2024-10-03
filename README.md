## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd 
df = pd.read_csv("Encoding Data.csv")
df
```
![o1](output/image.png)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![o2](<output/image copy.png>)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![o3](<output/image copy 2.png>)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![o4](<output/image copy 3.png>)
```
from sklearn.preprocessing import OneHotEncoder
ohe= OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![o5](<output/image copy 4.png>)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![o6](<output/image copy 5.png>)
```
!pip install --upgrade category_encoders
```
![o7](<output/image copy 6.png>)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![o8](<output/image copy 7.png>)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![o9](<output/image copy 8.png>)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![010](<output/image copy 9.png>)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
```
![o11](<output/image copy 10.png>)
```
df.skew()
```
![o12](<output/image copy 11.png>)
```
np.log(df["Highly Positive Skew"])
```
![o13](<output/image copy 12.png>)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![o14](<output/image copy 13.png>)
```
np.sqrt(df["Highly Positive Skew"])
```
![o15](<output/image copy 14.png>)
```
np.square(df["Highly Positive Skew"])
```
![o16](<output/image copy 15.png>)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
```
![o17](<output/image copy 16.png>)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
![o18](<output/image copy 17.png>)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![o19](<output/image copy 18.png>)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![o20](<output/image copy 19.png>)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![021](<output/image copy 20.png>)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![o22](<output/image copy 21.png>)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![o23](<output/image copy 22.png>)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![o24](<output/image copy 23.png>)
# RESULT:
  Thus the program to read the given data and perform Feature Encoding and Transformation process and save the data to a file is successfully executed

       
