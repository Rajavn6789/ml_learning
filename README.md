# Frontend Developer → Machine Learning Roadmap

You already know how to build things, think in components, handle async data, and work with APIs. That's a bigger head start than you think. This roadmap is tailored to your background — it skips what you already know, slows down where frontend intuition doesn't transfer, and frames new concepts in terms you're familiar with.

**Estimated time:** 8–10 months of consistent learning (1–2 hours/day)

---

## How to read this roadmap

- Each level is a milestone. Don't move on until you can explain the concepts without notes.
- Levels marked **[FAST]** can be completed in days, not weeks — you have prior knowledge.
- Levels marked **[SLOW]** need real time investment. Don't rush them.
- Practice exercises matter more than watching videos. Build things.

---

## Level 0 — Basic Setup & Mindset `[FAST]`

**a. Goal:** Understand what ML is and set up your environment.

**b. What ML actually is (plain English):**

Instead of writing rules (`if price > 500, classify as expensive`), you feed data to an algorithm and it _learns_ the rules itself. Your job as an ML engineer is to prepare the data, choose the right algorithm, train it, and evaluate how well it learned.

**c. AI vs ML vs Deep Learning:**

- **AI** — broad term: any machine that mimics human intelligence
- **ML** — a subset of AI: systems that learn from data
- **Deep Learning** — a subset of ML: uses neural networks with many layers; powers image recognition, LLMs, etc.

**d. Types of ML:**

| Type          | What it does                                    | Example               |
| ------------- | ----------------------------------------------- | --------------------- |
| Supervised    | Learns from labeled data (input → known output) | Spam detection        |
| Unsupervised  | Finds patterns in unlabeled data                | Customer segmentation |
| Reinforcement | Learns by trial and error with rewards          | Game-playing AI       |

**e. How an ML project works:**

1. Collect data
2. Clean and prepare data
3. Choose a model
4. Train the model
5. Evaluate performance
6. Deploy

**f. Tools to install:**

- Python 3.10+
- VS Code with the Python extension
- Jupyter Notebook (or use Google Colab — free, no setup)
- `pip install numpy pandas matplotlib seaborn scikit-learn`

---

## Level 1 — Python for JS Developers `[FAST — 1–2 weeks]`

**a. Goal:** Get productive in Python. Not learn programming from scratch — just learn the Python dialect.

**b. The mental shift:** Python reads like pseudocode. If you can read it, you can mostly write it.

**c. Key syntax differences from JavaScript:**

| JavaScript            | Python                          |
| --------------------- | ------------------------------- |
| `{}` blocks           | Indentation                     |
| `null`                | `None`                          |
| `true` / `false`      | `True` / `False`                |
| `===`                 | `==`                            |
| `const arr = [1,2,3]` | `arr = [1, 2, 3]`               |
| `arr.map(x => x*2)`   | `[x*2 for x in arr]`            |
| `async/await`         | `async/await` (same concept)    |
| `import x from 'y'`   | `import y` or `from y import x` |

**d. Skip or skim (you already know these):**

- Variables and data types
- Conditions and loops
- Functions
- OOP basics
- Error handling

**e. Spend real time on (Python-specific for ML):**

**NumPy** — the foundation of ML in Python. Think of it as typed arrays on steroids.

```python
import numpy as np

a = np.array([1, 2, 3])          # like a JS typed array
b = np.array([4, 5, 6])
print(a + b)                      # [5, 7, 9] — element-wise, no loop needed
print(a * 2)                      # [2, 4, 6] — broadcasting
matrix = np.zeros((3, 3))         # 3x3 matrix of zeros
```

**Pandas** — like working with spreadsheets or SQL tables in code.

```python
import pandas as pd

df = pd.read_csv('data.csv')        # load a dataset
df.head()                           # first 5 rows
df['age'].mean()                    # average age
df[df['age'] > 30]                  # filter rows
df.groupby('city')['salary'].mean() # group and aggregate
```

**Matplotlib** — charting, like D3 but imperative.

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

**f. Practice projects:**

- Load a public CSV (e.g. Titanic dataset), filter rows, compute summary stats, plot a histogram
- Write a list comprehension that squares all even numbers in a list

---

## Level 2 — Mathematics for ML `[SLOW — 4–6 weeks]`

**a. Goal:** Build numerical intuition. This is the hardest level coming from frontend. CSS and DOM give you visual, spatial intuition. ML requires numerical, algebraic intuition. Don't skip or rush this.

**b. The honest truth:** You won't need to derive equations from scratch. But you need to understand what's happening well enough to debug models and make good decisions.

---

### c. Linear Algebra

**Why it matters:** Every ML model stores data as numbers in grids (matrices). A neural network is literally just matrix multiplications stacked on top of each other. If you don't understand matrices, you're guessing at what's happening inside your model.

**i. Vectors**

A vector is an ordered list of numbers. In ML, it represents one data point.

```text
user = [age, income, num_purchases] = [28, 55000, 12]
```

Think of it as a point in multi-dimensional space. The more features, the more dimensions.

**ii. Matrices**

A matrix is a 2D grid of numbers. Your entire dataset is a matrix — each row is one sample, each column is one feature.

```text
Dataset (4 users × 3 features):
[[28, 55000, 12],
 [34, 72000,  5],
 [22, 41000, 20],
 [45, 90000,  3]]
```

**iii. Matrix Multiplication**

How a model transforms input data into predictions. If input is shape `(100, 3)` and weight matrix is `(3, 1)`, the output is `(100, 1)` — one prediction per user.

```python
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))  # matrix multiply
```

**iv. Dot Product**

Measures similarity between two vectors. Used in recommendation systems, search, and attention mechanisms.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))  # 1*4 + 2*5 + 3*6 = 32
```

**v. Eigenvalues / Eigenvectors**

Used in PCA (Principal Component Analysis) to find the most important directions in your data. You don't need to compute them manually — just understand they tell you _where the variance is_.

**vi. Practice:**

```python
# Implement matrix multiply manually, then verify
A = [[1,2],[3,4]]
B = [[5,6],[7,8]]
result = [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
print(result)                  # compare with np.dot(A, B)
```

---

### d. Statistics & Probability

**Why it matters:** ML models don't give you certainty — they give you probabilities. "This email is 92% likely to be spam." You need to understand how to interpret and evaluate those numbers.

**i. Descriptive Stats**

| Concept       | What it means                     | Python            |
| ------------- | --------------------------------- | ----------------- |
| Mean          | Average value                     | `np.mean(data)`   |
| Median        | Middle value (robust to outliers) | `np.median(data)` |
| Variance      | How spread out data is            | `np.var(data)`    |
| Std deviation | Square root of variance           | `np.std(data)`    |

**ii. Distributions**

- **Normal (Gaussian):** Bell curve. Most natural data (heights, errors) follows this.
- **Uniform:** Every value equally likely.
- **Skewed:** Tail on one side — common in income or price data.

Understanding distribution shape tells you how to preprocess your data.

**iii. Probability Basics**

- `P(A)` — probability of event A happening
- `P(A|B)` — probability of A _given_ B happened (conditional probability)
- **Bayes' Theorem:** `P(A|B) = P(B|A) * P(A) / P(B)` — the foundation of Naive Bayes classifiers

**iv. Correlation**

How much two variables move together. Range: -1 to 1.

- `1.0` = perfectly correlated (price goes up, demand goes up)
- `-1.0` = perfectly inverse (temperature up, hot chocolate sales down)
- `0` = no relationship

```python
df[['age', 'salary']].corr()  # correlation matrix
```

**v. Practice:** Load the Titanic dataset. Compute survival rate by gender, age distribution of survivors, and correlation between fare and survival.

---

### e. Calculus

**Why it matters:** Training a model means adjusting its internal numbers (weights) to reduce prediction error. That process — gradient descent — is pure calculus.

**i. Derivatives**

A derivative measures how much the output changes when the input changes slightly. In ML: "if I increase this weight by 0.001, how much does my error change?"

```text
f(x) = x²
f'(x) = 2x       ← derivative
f'(3) = 6        ← at x=3, increasing x by 1 increases f by ~6
```

**ii. Gradients**

A gradient is a vector of partial derivatives — one for each parameter in your model. It points in the direction of steepest increase of the loss.

**iii. Gradient Descent**

The core training algorithm. Move each weight a small step in the _opposite_ direction of the gradient (to reduce loss).

```python
# Minimize f(x) = x² using gradient descent
x = 10.0          # start somewhere
lr = 0.1          # learning rate — how big a step to take

for i in range(50):
    gradient = 2 * x     # derivative of x²
    x = x - lr * gradient
    print(f"x={x:.4f}, f(x)={x**2:.4f}")
# x converges toward 0
```

**iv. Chain Rule**

Used in backpropagation to compute gradients through many layers. Conceptually: if `z = f(g(x))`, then `dz/dx = f'(g(x)) * g'(x)`. You won't derive it by hand in practice — but PyTorch's autograd does this for you automatically.

**v. Practice:** Implement gradient descent from scratch to find the minimum of `f(x) = (x-3)²`. The answer should converge to `x = 3`.

---

### f. Recommended Resources (free)

- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [StatQuest with Josh Starmer — Statistics](https://www.youtube.com/@statquest)

---

## Level 3 — Data Handling `[MEDIUM — 2–3 weeks]`

**a. Goal:** Work with real, messy datasets. Real data is never clean.

**b. The frontend parallel:** This is like normalizing API responses before passing them to your components — except here the "component" is a model, and bad data ruins predictions instead of rendering.

**c. Missing values:**

```python
df.isnull().sum()                  # count missing per column
df['age'].fillna(df['age'].mean()) # fill with mean
df.dropna()                        # drop rows with any missing value
```

**d. Outliers:**

```python
# Values more than 3 std deviations from mean are likely outliers
mean, std = df['salary'].mean(), df['salary'].std()
df = df[df['salary'].between(mean - 3*std, mean + 3*std)]
```

**e. Encoding categories:**

ML models only understand numbers. Text labels must be converted.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])  # "London" → 0, "Paris" → 1
```

**f. Feature scaling:**

Different scales (age: 0–100, salary: 0–200000) confuse distance-based models. Normalize them.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])
```

**g. Projects:**

- Download the [Titanic dataset](https://www.kaggle.com/c/titanic/data), clean it end-to-end, and do a full EDA
- Download a sales CSV, handle missing values and outliers, plot key trends

---

## Level 4 — Machine Learning Fundamentals `[MEDIUM — 2 weeks]`

**a. Goal:** Understand how models are trained and evaluated before writing a single model.

**b. Dataset splitting:**

You never train and evaluate on the same data — that's like testing your own code with examples you wrote. You need unseen data to know if the model actually learned something general.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 80% for training, 20% for testing
```

**c. Overfitting vs Underfitting:**

| Problem      | What it means                                    | Frontend analogy                                    |
| ------------ | ------------------------------------------------ | --------------------------------------------------- |
| Overfitting  | Model memorized training data, fails on new data | Hardcoded pixel values instead of responsive design |
| Underfitting | Model too simple to capture patterns             | Using only one breakpoint for all screen sizes      |

**d. Evaluation metrics for regression (predicting a number):**

- **MAE** — average of absolute errors. Easy to interpret ("off by $500 on average")
- **RMSE** — punishes large errors more. Useful when big mistakes are costly
- **R²** — how much variance your model explains. 1.0 = perfect, 0 = no better than guessing the mean

**e. Evaluation metrics for classification (predicting a category):**

- **Accuracy** — % of correct predictions. Misleading on imbalanced data
- **Precision** — of all predicted positives, how many were actually positive? (avoid false alarms)
- **Recall** — of all actual positives, how many did you catch? (avoid missing real cases)
- **F1-score** — harmonic mean of precision and recall. Use when both matter
- **ROC-AUC** — overall model quality across all thresholds

---

## Level 5 — Supervised Learning `[MEDIUM — 4–6 weeks]`

**a. Goal:** Build your first real prediction models.

### b. Regression

Predict a continuous number (price, score, temperature).

**i. Linear Regression** — draws the best-fit line through your data.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**ii. Regularization** — prevents overfitting by penalizing large weights:

- **Ridge (L2):** shrinks all weights toward zero
- **Lasso (L1):** can zero out irrelevant features entirely (automatic feature selection)

**iii. Projects:**

- House price prediction (use Kaggle's California Housing dataset)
- Salary prediction based on experience and role

### c. Classification

Predict a category (spam/not spam, churn/stay).

**i. Logistic Regression** — despite the name, it's a classifier. Outputs a probability between 0 and 1.

**ii. Decision Tree** — a flowchart of yes/no questions. Interpretable but overfits easily.

**iii. Random Forest** — many decision trees voting together. More robust, harder to overfit.

**iv. K-Nearest Neighbors (KNN)** — classifies by looking at the K most similar training examples. Intuitive but slow on large datasets.

**v. Support Vector Machine (SVM)** — finds the widest possible boundary between classes.

**vi. Naive Bayes** — applies Bayes' theorem. Fast, good baseline for text classification.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```

**vii. Projects:**

- Spam email detector
- Customer churn prediction (will this user cancel their subscription?)

---

## Level 6 — Unsupervised Learning `[MEDIUM — 2–3 weeks]`

**a. Goal:** Find patterns when you don't have labels.

**b. The difference:** In supervised learning, every training example has a correct answer. In unsupervised learning, you just have data — the model finds structure on its own.

### c. Clustering

Group similar data points together without being told what the groups are.

**i. K-Means:**

```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X)
labels = model.labels_   # which cluster each point belongs to
```

**ii. Choosing K:** use the **elbow method** — plot inertia vs K and look for the bend.

**iii. Projects:**

- Customer segmentation (group users by behaviour without predefined categories)

### d. Dimensionality Reduction

Reduce many features to 2–3 for visualization or to remove noise.

**i. PCA** — finds the axes of maximum variance in your data and projects onto them.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)   # now visualizable in 2D
```

**ii. t-SNE** — better for visualization, not for downstream ML. Use it to explore data.

---

## Level 7 — Feature Engineering `[MEDIUM — 2–3 weeks]`

**a. Goal:** Improve model performance by improving your inputs, not your model.

**b. The insight:** A simple model with great features beats a complex model with bad features. This is where domain knowledge pays off.

**c. Creating new features:**

```python
df['age_income_ratio'] = df['age'] / df['income']    # interaction feature
df['is_senior'] = (df['age'] > 60).astype(int)       # binning
df['log_salary'] = np.log1p(df['salary'])             # log-transform skewed data
```

**d. Feature importance** (which features actually matter):

```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
```

**e. Handling imbalanced data** (e.g. 99% not-fraud, 1% fraud):

- **Oversample** the minority class: `imblearn.over_sampling.SMOTE`
- **Undersample** the majority class
- Use `class_weight='balanced'` in sklearn models

---

## Level 8 — Advanced ML `[MEDIUM — 3–4 weeks]`

**a. Goal:** Use the models that win Kaggle competitions and power production systems.

### b. Ensemble Learning

Combine multiple models to get better predictions than any single model.

**i. Gradient Boosting** — builds trees sequentially, each one correcting the errors of the last.

- **XGBoost** — fast, regularized gradient boosting. The go-to for tabular data.
- **LightGBM** — faster than XGBoost on large datasets.
- **CatBoost** — handles categorical features natively.

```python
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)
```

### c. Hyperparameter Tuning

Model parameters you set before training (not learned from data).

**i. Grid Search:**

```python
from sklearn.model_selection import GridSearchCV
params = {'max_depth': [3, 5, 7], 'n_estimators': [100, 200]}
grid = GridSearchCV(RandomForestClassifier(), params, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

**ii. Cross-validation** — split data into K folds, train on K-1, test on 1, rotate. More reliable than a single train/test split.

---

## Level 9 — Deep Learning `[SLOW — 6–8 weeks]`

**a. Goal:** Understand neural networks and build models for images and text.

**b. The mental model:** A neural network is a function that maps inputs to outputs through many layers of linear transformations + non-linear activations. Training adjusts the weights using gradient descent (Level 2 calculus pays off here).

**c. Start with PyTorch** (more Pythonic, better for learning internals) over TensorFlow.

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)   # 10 inputs → 64 neurons
        self.fc2 = nn.Linear(64, 1)    # 64 → 1 output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

### d. CNN (Images)

**i.** Convolution = sliding a filter over an image to detect edges, textures, patterns.

```python
self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
self.pool = nn.MaxPool2d(2, 2)
```

**ii. Projects:** Image classifier (cats vs dogs), MNIST digit recognition

### e. RNN / Transformers (Text & Sequences)

**i. RNN/LSTM** — processes sequences step by step. Remembers context.

**ii. Transformers** — processes all tokens in parallel using attention. Powers all modern LLMs.

**iii. Projects:** Sentiment analysis, text classification

---

## Level 10 — Natural Language Processing (NLP) `[MEDIUM — 3–4 weeks]`

**a. Goal:** Work with text data end-to-end.

**b. Text preprocessing pipeline:**

```python
import re
text = "Hello, World! This is NLP."
text = text.lower()                          # lowercase
text = re.sub(r'[^\w\s]', '', text)          # remove punctuation
tokens = text.split()                        # tokenize
# → ['hello', 'world', 'this', 'is', 'nlp']
```

**c. Word Embeddings** — represent words as vectors where similar words are close together.

```python
from gensim.models import Word2Vec
# "king" - "man" + "woman" ≈ "queen"
```

**d. Using BERT/Transformers** (the practical approach today):

```python
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
classifier("This movie was fantastic!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

**e. Projects:**

- Movie review sentiment classifier
- Text summarizer using HuggingFace

---

## Level 11 — Computer Vision `[MEDIUM — 3–4 weeks]`

**a. Goal:** Build models that understand images.

**b. Key skill:** Learn to use pretrained models (transfer learning) instead of training from scratch.

```python
import torchvision.models as models
model = models.resnet50(pretrained=True)

# Freeze all layers except the last
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your task (e.g. 2 classes)
model.fc = nn.Linear(model.fc.in_features, 2)
```

**c. Projects:**

- Image classifier using transfer learning (ResNet / EfficientNet)
- Object detection with YOLO (pretrained)

---

## Level 12 — MLOps `[MEDIUM — 3–4 weeks]`

**a. Goal:** Deploy models as real services. This level will feel familiar — it's backend/DevOps territory.

**b. You already know:**

- APIs (REST)
- Deployment concepts
- Environment variables and config

**c. Serving a model as an API:**

```python
from fastapi import FastAPI
import pickle, numpy as np

app = FastAPI()
model = pickle.load(open('model.pkl', 'rb'))

@app.post('/predict')
def predict(data: dict):
    features = np.array(data['features']).reshape(1, -1)
    return {'prediction': int(model.predict(features)[0])}
```

**d. Containerizing with Docker:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**e. ML-specific concerns:**

- Model versioning (MLflow)
- Data drift — model degrades when real-world data shifts from training data
- Retraining pipelines

---

## Level 13 — Advanced AI Topics `[SLOW — ongoing]`

**a. Goal:** Stay current with the state of the art.

**b. Topics:**

- **Reinforcement Learning** — agents learning from reward signals (games, robotics)
- **Generative AI** — GANs, diffusion models, image generation
- **Large Language Models** — how GPT/Claude work at a high level
- **Prompt Engineering** — getting reliable outputs from LLMs
- **RAG (Retrieval-Augmented Generation)** — combining LLMs with your own data
- **AI Agents** — LLMs that use tools and make multi-step decisions

---

## Level 14 — Portfolio Projects

Build these in order. Each one demonstrates a complete ML workflow.

### a. Beginner

- **House price predictor** — regression, feature engineering, sklearn
- **Spam classifier** — text classification, Naive Bayes or Logistic Regression
- **Movie recommender** — collaborative filtering, cosine similarity

### b. Intermediate

- **Fraud detection** — imbalanced classification, XGBoost, threshold tuning
- **Customer churn prediction** — binary classification, SHAP for explainability
- **Recommendation system** — matrix factorization or content-based filtering

### c. Advanced

- **Chatbot with LLM + RAG** — FastAPI + LangChain + vector database
- **Image classifier web app** — PyTorch model served via FastAPI, React frontend (your strength!)
- **End-to-end ML pipeline** — data ingestion → training → serving → monitoring

> **Tip for your portfolio:** The image classifier web app plays to both your strengths — you own the frontend, and you built the ML backend. That's a compelling story for a hiring manager.

---

## Revised Timeline for Frontend Developers

| Period      | Focus          | Notes                                    |
| ----------- | -------------- | ---------------------------------------- |
| Weeks 1–2   | Level 0–1      | Fast — you already code                  |
| Weeks 3–8   | Level 2 (Math) | The real investment. Don't rush.         |
| Weeks 9–14  | Levels 3–4     | Data handling and ML fundamentals        |
| Months 4–6  | Levels 5–7     | Core ML algorithms + feature engineering |
| Months 6–8  | Levels 8–9     | Advanced ML + intro to deep learning     |
| Months 8–10 | Levels 10–12   | Specialization + deployment              |
| Month 10+   | Levels 13–14   | Advanced topics + portfolio              |

---

## Recommended Learning Order

Python → Math → Data → ML Fundamentals → Supervised ML → Unsupervised ML → Feature Engineering → Advanced ML → Deep Learning → NLP/CV → MLOps → Advanced AI → Portfolio

---

## Resources

| Topic                   | Resource                                                                                                                    | Cost |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---- |
| Python for Data Science | [Kaggle Learn — Python](https://www.kaggle.com/learn/python)                                                                | Free |
| Linear Algebra          | [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)         | Free |
| Calculus                | [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)               | Free |
| Statistics              | [StatQuest with Josh Starmer](https://www.youtube.com/@statquest)                                                           | Free |
| ML fundamentals         | [fast.ai — Practical Deep Learning](https://course.fast.ai)                                                                 | Free |
| Hands-on practice       | [Kaggle competitions](https://www.kaggle.com/competitions)                                                                  | Free |
| Deep Learning           | [Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | Free |
| Transformers/NLP        | [HuggingFace Course](https://huggingface.co/learn/nlp-course)                                                               | Free |
