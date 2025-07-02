## Project 2: *Fake News Detection Using Word Frequencies*

### 🧠 Motivation

Distinguishing real from fake news is a socially relevant and technically challenging task. Unlike images or tabular data, raw text must be transformed into numerical representations before models can learn from it. This project focuses on that critical first step: **text preprocessing and feature engineering**.

Students will explore how different ways of representing text—such as **bag-of-words** and **TF-IDF**—impact model performance, interpretability, and generalization. The goal is not just to train a classifier, but to understand how the **quality of feature extraction** shapes learning outcomes.

### 🎯 Objectives

- Preprocess raw text and construct interpretable numerical features  
- Extract document-level representations using **TF-IDF** or bag-of-words  
- Train and evaluate classical ML models on these features  
- Interpret which words or n-grams are most important for distinguishing fake vs. real news  
- Optionally explore neural models or word embeddings for comparison

### 📦 Dataset

- **Fake and Real News Dataset** ([Kaggle link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset))  
  Labeled news articles from 2016–2017 with titles, full body text, and metadata.

You may choose to work with:
- Titles only  
- Full article body text  
- Both (to compare predictive value)

### 📐 TF-IDF: Term Frequency–Inverse Document Frequency

Let $D = \{d_1, d_2, \dots, d_n\}$ be the collection of documents.

For a term $t$ and a document $d$, define:

- **Term Frequency (TF)**:

$$
\text{tf}(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total number of terms in } d}
$$


- **Inverse Document Frequency (IDF)**:

$$
\text{idf}(t, D) = \log\left(\frac{n}{1 + |\{d_i \in D : t \in d_i\}|}\right)
$$

Then the **TF-IDF score** for term $t$ in document $d$ is:
  $$
  \text{tfidf}(t, d, D) = \text{tf}(t, d) \cdot \text{idf}(t, D)
  $$

This scoring favors terms that are **frequent within a document** but **rare across the corpus**, helping the model emphasize words that carry discriminative power.

### 🛠️ Suggested Tools and Models

#### 🔹 Preprocessing and Feature Extraction

- You can use `TfidfVectorizer` or `CountVectorizer` from `sklearn.feature_extraction.text`: these handle tokenization, stopword removal, lowercasing, and optional n-gram extraction.
- **Optional challenge:** implement a basic tokenizer using `re` or `nltk.word_tokenize()` to understand how tokenization choices affect feature space.

#### 🔹 Suggested Models

- Logistic regression  
- Linear SVM  
- Optional: MLP baseline for raw TF-IDF vectors  
- Optional: simple RNN/CNN for word sequences (extension only)

### ✅ Minimum Requirements

- Preprocess the dataset:  
  - Lowercase text, remove punctuation, and tokenize  
  - Optionally remove stopwords and apply stemming or lemmatization

- Compute document representations using one of:  
  - Count vector (bag-of-words)  
  - TF-IDF features

- Train and evaluate at least two classification models (e.g., logistic regression, SVM)

- Report and visualize:
  - Accuracy, precision, recall, F1 score  
  - Confusion matrix  
  - Top weighted words or n-grams learned by the model

### 🚀 Optional Extensions

- Compare multiple feature types: bag-of-words vs. TF-IDF vs. embeddings  
- Use n-grams (e.g., bigrams or trigrams) to capture short phrases  
- Visualize learned word importance (e.g., using word clouds or sorted weights)  
- Evaluate model robustness to article length, publication date, or other metadata  
- Use dimensionality reduction (e.g., PCA or t-SNE) to visualize document clusters

### 📚 References

- Jurafsky & Martin (2024) – [*Speech and Language Processing* (Ch. 6: Vector Semantics and Embeddings)](https://web.stanford.edu/~jurafsky/slp3/6.pdf)  
- Zhou & Zafarani (2019) – [*A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities*](https://arxiv.org/pdf/1812.00315)  
- scikit-learn documentation – [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)  
- nltk – [Natural Language Toolkit](https://www.nltk.org/)