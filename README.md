# Movie Review Sentiment Analysis

This project implements an automated sentiment classification pipeline to analyze IMDB movie reviews using **LSTM** and **Transformer** architectures. By leveraging pre-trained word embeddings like **GloVe** and **ConceptNet Numberbatch**, the system distinguishes between positive and negative viewer feedback to enable large-scale opinion mining for media platforms.

---

### Project Overview
The primary goal of this [Movie Review Sentiment Analysis](https://github.com/KarrieB26/Movie-Review-Sentiment-Analysis/blob/main/movie_review_sentiment_analysis.ipynb) project is to build a robust predictive model that quantifies audience sentiment from unstructured text. Key objectives include:
* **Comparative Modeling:** Evaluating the performance of Bidirectional LSTMs against Multi-Head Self-Attention Transformer models.
* **Knowledge-Enhanced Embeddings:** Comparing standard GloVe embeddings with ConceptNet Numberbatch to capture deeper semantic relationships.
* **Business-Driven Metrics:** Prioritizing **F1-Score** and **Recall** to minimize "missed revenue" opportunities from incorrectly flagged positive reviews.

---

### Key Technical Features
* **Text Preprocessing:** Automated cleaning of raw HTML tags and special characters using Regex, followed by Keras-based tokenization and sequence padding.
* **Architecture Optimization:** Includes a tuned LSTM with 128 units and a stacked-attention Transformer to capture complex dependencies in narrative text.
* **Performance Analysis:** Utilizes **PCA (Principal Component Analysis)** to visualize how different embedding spaces (GloVe vs. ConceptNet) cluster sentiment-bearing words.

---

### Technical Stack
* **Languages:** Python
* **Deep Learning:** TensorFlow, Keras
* **Data Science:** Pandas, NumPy, Scikit-learn, Matplotlib, SciPy
* **Embeddings:** [GloVe (100d)](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt), [ConceptNet Numberbatch (300d)](https://github.com/commonsense/conceptnet-numberbatch)

---

### Getting Started
1. **Clone the repository:**
   ```bash
   git clone https://github.com/KarrieB26/Movie-Review-Sentiment-Analysis.git
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook:**
   Execute `movie_review_sentiment_analysis.ipynb` to train the models and view the comparative results.

---

### Contributors
* **[Karrie Butcher](https://github.com/KarrieB26)**
* **Nicko Lomelin**
* **Thanh Tuan Pham**
