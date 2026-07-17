# Recommended 6-Week Learning Plan — v2

> An expanded version with more projects, mini-exercises, datasets, and tooling for each week.
> Each week has a **core project** plus **extra projects** to reinforce the concepts.

---

## Week 1 — Text Fundamentals

**Learn:**

- NLP basics & the NLP pipeline
- Tokenization (word, subword, sentence)
- Text cleaning (lowercasing, stopwords, stemming, lemmatization)
- Bag of Words
- TF-IDF
- N-grams & regex-based feature extraction

**Build:**

- ✅ Spam classifier (core)
- News topic categorizer (sports / politics / tech)
- Language detector
- Keyword extractor for articles
- Simple autocomplete using n-grams

**Tools & datasets:** `scikit-learn`, `nltk`, `spaCy` · SMS Spam Collection, 20 Newsgroups

**Stretch goal:** Compare Naive Bayes vs. Logistic Regression on the same dataset and explain the difference.

---

## Week 2 — Embeddings & Similarity

**Learn:**

- Embeddings (why dense vectors beat one-hot)
- Word2Vec (CBOW & Skip-gram)
- GloVe
- Cosine similarity & vector arithmetic
- Dimensionality reduction (PCA / t-SNE) for visualization

**Build:**

- ✅ Movie recommendation using embeddings (core)
- Semantic search over a document set
- "King − Man + Woman = ?" analogy explorer
- Duplicate question detector (Quora-style)
- Word cluster visualizer (t-SNE plot)

**Tools & datasets:** `gensim`, `scikit-learn`, `matplotlib` · Google News vectors, GloVe pretrained, MovieLens

**Stretch goal:** Train your own Word2Vec model on a custom corpus and inspect nearest neighbors.

---

## Week 3 — Neural Networks for NLP

**Learn:**

- Neural networks for NLP (feed-forward, RNN, LSTM, GRU)
- Attention mechanism (intuition + math)
- Sequence models & padding/batching
- Embedding layers

**Build:**

- ✅ Sentiment analysis (core, LSTM-based)
- Named Entity Recognition (NER) tagger
- Text generator (character-level RNN)
- POS (part-of-speech) tagger
- Emoji predictor from text

**Tools & datasets:** `PyTorch` or `TensorFlow/Keras` · IMDB reviews, CoNLL-2003, Twitter sentiment

**Stretch goal:** Add an attention layer to your sentiment model and visualize which words it focuses on.

---

## Week 4 — Transformers

**Learn:**

- Transformers architecture (self-attention, multi-head attention)
- Encoder
- Decoder
- Positional encoding
- BERT & the family of pretrained encoders

**Build:**

- ✅ Question answering bot (core)
- Text summarizer (extractive)
- Masked-word prediction demo
- Named entity extraction with BERT
- Zero-shot text classifier

**Tools & datasets:** `transformers` (Hugging Face), `datasets` · SQuAD, CNN/DailyMail

**Stretch goal:** Implement scaled dot-product attention from scratch in NumPy and compare to the library version.

---

## Week 5 — Fine-Tuning Pretrained Models

**Learn:**

- Fine-tuning pretrained models (full vs. parameter-efficient / LoRA)
- Hugging Face ecosystem (`Trainer`, `Datasets`, `Hub`)
- Transfer learning best practices
- Handling class imbalance & overfitting

**Build:**

- ✅ Text classifier (core, fine-tuned BERT)
- Custom domain intent classifier (e.g., customer support)
- Toxic comment detector
- Resume ↔ job description matcher
- Fine-tuned summarizer (abstractive)

**Tools & datasets:** Hugging Face `Trainer`, `peft`, `accelerate` · Jigsaw Toxic Comments, custom CSV

**Stretch goal:** Push your fine-tuned model to the Hugging Face Hub and load it back for inference.

---

## Week 6 — Prompting, Evaluation & Deployment

**Learn:**

- Prompt engineering (zero-shot, few-shot, chain-of-thought)
- Retrieval-Augmented Generation (RAG) basics
- Evaluation (accuracy, F1, BLEU, ROUGE, human eval)
- Deployment (REST API, containerization, monitoring)

**Build:**

- ✅ Complete NLP application (core, end-to-end)
- RAG-powered document Q&A over your own PDFs
- Chatbot with conversation memory
- Streamlit/Gradio demo UI for any earlier model
- Model evaluation dashboard

**Tools & datasets:** `FastAPI`, `Streamlit`/`Gradio`, `Docker`, a vector DB (`FAISS`/`Chroma`), an LLM API · your own data

**Stretch goal:** Deploy your app to a cloud host and add basic logging + latency monitoring.

---

## Capstone Project Ideas

Combine multiple weeks into one polished portfolio piece:

- **Customer support assistant** — intent classification + RAG + chat memory
- **Research paper explainer** — summarization + Q&A over PDFs
- **Content moderation pipeline** — toxicity + spam + language detection
- **Personal knowledge search engine** — embeddings + semantic search + UI
- **Multilingual sentiment dashboard** — language detection + sentiment + charts

---

## Skills Checklist

By the end of this phase, you should be comfortable building applications similar to:

- Chatbots
- Search engines
- Document summarizers
- AI assistants
- Sentiment analysis systems
- Question answering systems
- RAG applications
- Content moderation / classification systems

**The key progression is:**

> Text → Tokens → Embeddings → Attention → Transformers → Fine-Tuning → Prompting/RAG → Deployed NLP Applications

---

## Suggested Weekly Rhythm

| Days | Focus |
| --- | --- |
| Mon–Tue | Learn concepts + read/watch |
| Wed–Thu | Build the core project |
| Fri | Try one extra project |
| Sat | Stretch goal / experiment |
| Sun | Review, write notes, refactor code |
