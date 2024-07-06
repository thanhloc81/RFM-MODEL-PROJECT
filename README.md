# RFM-MODEL-PROJECT
 Build a flow to deploy Segmentation evaluation through Python programming.

 ## Content
RFM (Recency – Frequency – Monetary): is a part of Marketing Analysis and is used to analyze customer value, thereby helping businesses analyze each group of customers they have. From there, there are marketing campaigns or special care.
 ## Context
SuperStore Company is a **global retail company** - Global. So the company has many customers.
On the occasion of Christmas and New Year, the Marketing Department wants to **run marketing campaigns** to thank customers who have supported the company over the past time. As well as exploiting customers who have the potential to become loyal customers. However, the Marketing Department has not yet been able to group each customer this year because the data set is too large to be processed manually like in previous years, so we asked the Data Analysis Department to assist in implementing a classification problem. Segment each customer to deploy each marketing program suitable for each customer group.

The Marketing Director also proposed using **the RFM model**, but in the past when the company was small, the team could calculate and classify it themselves using Excel. Currently, the amount of data is too large, so we want the Data Department to build a flow to deploy Segmentation evaluation through Python programming.

## Step
### 1. Explore Data Analysis (EDA)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load data

df = pd.read_excel('/content/ecommerce retail.xlsx', sheet_name='ecommerce retail')
print(df.head())

# Check null & data type
df.info()
```
![image](https://github.com/thanhloc81/RFM-MODEL-PROJECT/assets/151768013/0027a43b-a369-4644-8227-1a35e55d260e)
```python
df.describe()
```
![image](https://github.com/thanhloc81/RFM-MODEL-PROJECT/assets/151768013/fef6277c-5396-4a88-aba0-b3943d024d05)

We have a quick look at the first data set, the start date is **12/01/2010** and the end date is **12/09/2011**. However, the Quantity and UnitPrice columns are not correct because the smallest value is still negative. We will normalize this column below and this is some issues that should be solved:
- CustomerID have null
- Quantity < 0
- Unit price < 0
- UK transactions have nulls
- The main columns that we should get: InvoiceNo, InvoiceDate, UnitPrice, CustomerID, (StockCode, Description)

**Clean data**
```python
# Delete row have CustomerID null
df_copy = df.copy()
df_copy = df_copy.dropna(subset=['CustomerID'])

# Delete row have Transaction Cancle
df_copy['InvoiceNo'] = df_copy['InvoiceNo'].apply(str)
df_copy['CustomerID'] = df_copy['CustomerID'].apply(str)
df_copy = df_copy.drop(df_copy[df_copy['InvoiceNo'].str.contains("C")].index)

# Convert UnitPrice and Quantity into positive values
df_copy = df_copy[df_copy['UnitPrice']>0]
df_copy =  df_copy[df_copy['Quantity']>0]

# Replace UnitPrice = 0 using Mean
mean_price = df_copy['UnitPrice'].mean()
df_copy['UnitPrice'] = df_copy['UnitPrice'].mask(df_copy['UnitPrice'] == 0, mean_price)
```
![image](https://github.com/thanhloc81/RFM-MODEL-PROJECT/assets/151768013/c8b929a8-b4c5-451e-b37f-74841a8e2d8b)
![image](https://github.com/thanhloc81/RFM-MODEL-PROJECT/assets/151768013/c5d2c1b4-cd87-4c9f-9721-d64160b3dd17)

### 2. Creating RFM score
```pthon
# Calculated Recency Score
max_date = df_copy['InvoiceDate'].max() + pd.Timedelta(days=21)
Recency = df_copy.groupby('CustomerID')['InvoiceDate'].max().reset_index()
Recency['InvoiceDate'] = (max_date - Recency['InvoiceDate']) // pd.Timedelta(days=1)
Recency = Recency.rename(columns = {'InvoiceDate':'Recency'})

# Calculated Frequency Score
Frequency = df_copy.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
Frequency = Frequency.rename(columns = {'InvoiceNo':'Frequency'})

# Calculated Monetary Score
df_copy['SubTotal'] = df_copy['Quantity']*df_copy['UnitPrice']
Monetary = df_copy.groupby('CustomerID')['SubTotal'].sum().reset_index()
Monetary = Monetary.rename(columns = {'SubTotal':'Monetary'})

# Create RFM Table
merge_df = pd.merge(pd.merge( Recency, Frequency, on='CustomerID'), Monetary, on='CustomerID')
print(merge_df)
```
![image](https://github.com/thanhloc81/RFM-MODEL-PROJECT/assets/151768013/956086ac-317b-4108-9af1-13ef6e134c2e)

```python
# Calculated rfm scored for every customer
merge_df['R_score'] = pd.qcut(merge_df['Recency'], q=5, labels=list(range(5, 0, -1)))
merge_df['f_score'] = pd.qcut(merge_df['Frequency'].rank(method='first'), q=5, labels= range(1,6))
merge_df['M_score'] = pd.qcut(merge_df['Monetary'], q=5, labels=range(1,6))
merge_df['rfm_score'] = merge_df.apply(lambda row: str(row['R_score']) + str(row['f_score']) + str(row['M_score']), axis=1)

# Create a table that will pair each customer with the appropriate segment.
df_rfm = pd.read_excel('/content/ecommerce retail.xlsx', sheet_name='Segmentation')

df_rfm['RFM Score'] = df_rfm['RFM Score'].str.split(',')
df_rfm = df_rfm.explode('RFM Score').reset_index(drop = True)
df_rfm['RFM Score'] = df_rfm['RFM Score'].str.strip()
df_rfm = df_rfm.rename(columns ={'RFM Score':'rfm_score'})
new_df=pd.merge(merge_df, df_rfm, how='left',on='rfm_score')
print(new_df)
```
![image](https://github.com/thanhloc81/RFM-MODEL-PROJECT/assets/151768013/c4a30232-e7ec-447c-ab3c-d6ed3ab7974b)

### 3. Visualizations and Insights
#### Overview
```python
# Overview
col_name = ['Recency','Frequency','Monetary']
fig,axes = plt.subplots(1,3, figsize=(20,5))

for i, col in enumerate(col_name):
  sns.distplot(new_df[col], ax =axes[i])
  axes[i].set_title('Distribute of %s' %col)
  mean_value = new_df[col].mean()
  axes[i].text(0.5, 0.95, f'Mean of {col}: {mean_value:.2f}', horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
plt.tight_layout()
plt.show()
```
![image](https://github.com/thanhloc81/RFM-MODEL-PROJECT/assets/151768013/1142d173-bcfe-4157-b4f8-7c6ffc053257)

- Of the three indicators, we see that Recency is the indicator that does not have too much difference between customer segments. **112 days** is the average value from the last purchase to the reporting date, and we see in the Recency chart that most new customers only made a purchase **21-100 days** before the reporting date. That shows the large purchasing power of customers in the last months of 2011.
- In the remaining two indicators, there will be quite a **big difference between customer segments** because we observe that the two Monetary and Frequency charts are quite left-skewed.
