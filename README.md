# 📧 Email Spam Detection using Machine Learning

## 📌 Overview

This project is a **Machine Learning-based Email Spam Detection System** that classifies emails as **Spam** or **Not Spam (Ham)**. Using supervised learning techniques, the model is trained on a dataset of labeled emails and achieves high accuracy in distinguishing between spam and legitimate messages.

---

## 🎯 Objectives

* Build a machine learning model to detect spam emails.
* Preprocess email text using NLP techniques.
* Train and evaluate the model on labeled data.
* Achieve high accuracy, precision, recall, and F1-score.

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Libraries & Tools:**

  * **scikit-learn** – Machine Learning algorithms
  * **pandas, numpy** – Data handling and preprocessing
  * **nltk** – Natural Language Processing (tokenization, stopwords removal)
  * **matplotlib, seaborn** – Data visualization
  * **Jupyter Notebook / Google Colab** – Development environment

---

## 📂 Dataset

* Dataset: [SpamAssassin / UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) or any labeled spam dataset.
* Contains two classes:

  * **Spam** → Unwanted/advertising/scam emails.
  * **Ham** → Legitimate emails.

---

## ⚙️ Methodology

1. **Data Collection**

   * Load dataset (CSV/TXT format).

2. **Data Preprocessing**

   * Lowercasing
   * Removing punctuation, numbers, and special characters
   * Stopwords removal
   * Tokenization
   * Stemming/Lemmatization
   * Convert text to numeric features using **TF-IDF / CountVectorizer**

3. **Model Training**

   * Algorithms tested:

     * Naïve Bayes
     * Logistic Regression
     * Support Vector Machine (SVM)
     * Random Forest
   * Best-performing model selected based on evaluation metrics.

4. **Model Evaluation**

   * Accuracy, Precision, Recall, F1-score
   * Confusion Matrix visualization

5. **Deployment (Optional)**

   * Save model using **pickle/joblib**
   * Create a simple Flask/Django web app for spam detection

---

## 📊 Results

* **Naïve Bayes Classifier** achieved **\~96% accuracy** on the test dataset.
* High precision and recall for spam detection, minimizing false positives.

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/email-spam-detection.git
   cd email-spam-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:

   ```bash
   jupyter notebook Spam_Detection.ipynb
   ```

4. (Optional) Run Flask app for prediction:

   ```bash
   python app.py
   ```

---

## 📈 Future Improvements

* Use **deep learning models** (LSTM, BERT) for better accuracy.
* Deploy as a **web app** with user interface.
* Extend to **multi-language spam detection**.
