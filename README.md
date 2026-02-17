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

# 🔹 TASK 4 – Text Feature Engineering

## 🎯 Objective

Convert text columns into a clean format for machine learning processing.

### Columns Used:

- Skills  
- Certifications  
- Job Role  

---

## 🧠 Step 1: Combine Text Columns

### Command Used:

```python
df['combined_text'] = df['Skills'] + " " + df['Certifications'] + " " + df['Job Role']
```

### 📌 Why We Did This:

Instead of vectorizing 3 separate columns, we:

- Combined them into one resume representation  
- Made it easier for TF-IDF to capture relationships  
- Simulated a real resume screening process  

---

## 🧠 Step 2: Text Cleaning

### Command Used:

```python
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['combined_text'] = df['combined_text'].apply(clean_text)
```

### 📌 Why We Cleaned Text:

- Convert to lowercase → "Python" and "python" treated the same  
- Remove special characters → remove noise  
- Remove extra spaces → improve consistency  

Without cleaning → TF-IDF creates duplicate features.

---

## ✅ Output After Cleaning

When we ran:

```python
df['combined_text'].sample(3)
```

We saw clean text like:

```
python machine learning aws certified data scientist
java spring backend developer
cybersecurity ethical hacking linux engineer
```

### Meaning:

✔ Cleaning worked  
✔ No commas  
✔ No uppercase  
✔ No unwanted symbols  

---

# 🔹 TASK 5 – Convert Text to Numerical (TF-IDF)

## 🎯 Objective

Machine learning models understand numbers, not words.

We converted text into numerical feature vectors.

---

## 🧠 Step 1: Apply TF-IDF

### Command Used:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X_text = tfidf.fit_transform(df['combined_text'])
```

### 📌 Why TF-IDF?

TF-IDF = **Term Frequency × Inverse Document Frequency**

It:

- Gives higher importance to rare but meaningful words  
- Reduces importance of common words  
- Highlights technical skills  
- Performs better than simple word counting  

---

## 🧠 Step 2: Check Shape

### Command Used:

```python
X_text.shape
```

### Output:

```
(1000, 28)
```

### 📌 What This Means

- 1000 resumes  
- 28 meaningful technical keywords extracted  

Your dataset had **28 strong resume features**.

That’s GOOD — not too noisy, not too large.

---

## 🧠 Step 3: Check Extracted Words

### Command Used:

```python
tfidf.get_feature_names_out()
```

### Output:

```
['ai', 'analyst', 'aws', 'certification', 'certified',
 'cybersecurity', 'data', 'deep', 'engineer', 'ethical',
 'google', 'hacking', 'java', 'learning', 'linux',
 'machine', 'ml', 'networking', 'nlp',
 'python', 'pytorch', 'react', 'researcher',
 'scientist', 'software', 'specialization',
 'sql', 'tensorflow']
```

---

## 📌 Why This Output Is Excellent

These are:

✔ Programming Skills → python, java, sql  
✔ AI Skills → machine, tensorflow, pytorch  
✔ Cybersecurity → hacking, ethical, networking  
✔ Job Roles → analyst, engineer, scientist  

This shows:

- Model is learning REAL hiring signals  
- Feature engineering is meaningful  
- Dataset cleaning worked perfectly  


