# AI-Hiring-Prediction-System
AI-powered resume screening system that predicts candidate hiring decisions using NLP feature engineering and machine learning models.
# ✅ TASK 1: Load and Understand the Dataset

## 🔹 1. Import Required Libraries

### Code Used:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
```

### Why:

- **pandas** → Data manipulation and analysis  
- **numpy** → Numerical operations  
- **seaborn & matplotlib** → Data visualization  
- **sklearn** → Machine learning models & preprocessing  

---

## 🔹 2. Load the Dataset

```python
df = pd.read_csv("AI-Based Hiring Prediction System.csv")
```

### Why:

`pd.read_csv()` is used to load CSV files into a Pandas DataFrame.

---

## 🔹 3. Display Sample Data

### First 5 Rows:

```python
df.head()
```

### Last 5 Rows:

```python
df.tail()
```

### Random Sample:

```python
df.sample(5)
```

### Insight:

- Dataset contains structured resume data.  
- Includes skills, certifications, education, salary expectations, etc.  
- **Recruiter Decision** appears to be the output label.  

---

## 🔹 4. Explanation

### ✔ Type of Data Present

The dataset contains:

### 📊 Numerical Data

- Experience (Years)  
- Salary Expectation ($)  
- Projects Count  
- AI Score (0–100)  

### 📝 Categorical/Text Data

- Skills  
- Education  
- Certifications  
- Job Role  
- Recruiter Decision  

This is a combination of structured and unstructured data.

---

### ✔ Target Column

**Recruiter Decision**

This is a binary classification target:

- Hire → 1  
- Reject → 0  

---

# ✅ TASK 2: Basic Data Inspection

## 🔹 Number of Rows and Columns

```python
df.shape
```

### Insight:

The dataset contains **1000 rows and 11 columns**.

---

## 🔹 Column Names

```python
df.columns
```

### Insight:

Provides overview of available features.

---

## 🔹 Data Types of Each Column

```python
df.info()
```

### Insight:

- 5 numerical columns (`int64`)  
- 6 object/text columns  
- Certifications has missing values  

---

## 🔹 Value Counts of Target

```python
df['Recruiter Decision'].value_counts()
```

### Insight:

- Confirms binary classification problem  
- Helps check for class imbalance  

---

## 🔹 Summary Statistics (Numerical Columns)

```python
df.describe()
```

### Insight:

- Shows mean, min, max, standard deviation  
- Salary expectations vary significantly  
- Experience ranges across different levels  
- Projects count varies among candidates  

---

## 🔹 Why Data Inspection Is Important

Before training a machine learning model:

- To detect missing values  
- To verify correct data types  
- To understand feature distribution  
- To detect class imbalance  
- To prevent errors during modeling  

Data inspection ensures data quality before building models.

---

# ✅ TASK 3: Data Cleaning and Preprocessing

## 🔹 1. Drop Unnecessary Columns

```python
df.drop(['Resume_ID', 'Name', 'AI Score (0-100)'], axis=1, inplace=True)
```

### Why:

- **Resume_ID** → Unique identifier (non-informative)  
- **Name** → Personal identifier, no predictive power  
- **AI Score** → Causes data leakage  

---

## 🔹 2. Convert Target Variable

```python
df['Recruiter Decision'] = df['Recruiter Decision'].map({
    'Hire': 1,
    'Reject': 0
})
```

### Why:

Machine learning models require numerical targets.

---

## 🔹 3. Check for Missing Values

```python
df.isnull().sum()
```

### Observation:

Certifications column contained missing values.

---

## 🔹 4. Handle Missing Values

```python
df['Certifications'] = df['Certifications'].fillna("No Certification")
```

### Why:

Since Certifications is text data used in NLP, replacing missing values with a placeholder maintains consistency.

---

## 🔹 5. Ensure Numeric Columns Are Correctly Formatted

```python
df.dtypes
```

All numeric columns are correctly stored as `int64`, so no conversion required.

