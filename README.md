# Amharic Sentiment Analysis Project

This project implements a sentiment analysis application for Amharic text using a **PyTorch-based backend** and a **React frontend**. The application allows users to input Amharic text and receive sentiment predictions categorized as **Positive**, **Neutral**, or **Negative**.  

It can be used for analyzing social media comments, reviews, or any Amharic text corpus.  

---

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Backend](#backend)  
3. [Frontend](#frontend)  
4. [Dataset](#dataset)  
5. [Model](#model)  
6. [Setup Instructions](#setup-instructions)  
7. [Usage](#usage)  
8. [Validation](#validation)  
9. [Contributing](#contributing)  
10. [License](#license)  

---

## Project Structure

```
amharic-sentiment-analysis
├── backend
│   ├── app.py
│   ├── model
│   │   └── sentiment_model.py
│   ├── utils
│   │   └── preprocess.py
│   ├── requirements.txt
│   └── README.md
├── frontend
│   ├── public
│   │   └── index.html
│   ├── src
│   │   ├── App.js
│   │   ├── components
│   │   │   └── SentimentForm.js
│   │   └── styles
│   │       └── App.css
│   ├── package.json
│   └── README.md
└── README.md
```
---

## Backend

- Built with **Flask** and **PyTorch**.  
- Handles preprocessing, tokenization, and model inference.  
- Key files:
  - `app.py`: Flask API endpoints  
  - `model/sentiment_model.py`: PyTorch LSTM model  
  - `utils/preprocess.py`: Preprocessing functions  
  - `requirements.txt`: Python dependencies  

---

## Frontend

- Built with **React**.
- Additional Features with **HTML**, **CSS** & **JS**
- Handles user input and displays predictions.  
- Key files:
  - `public/index.html`  
  - `src/App.js`  
  - `src/components/SentimentForm.js`  
  - `src/styles/App.css`  
  - `package.json`  

---

## Dataset

- `data/final.csv` contains Amharic sentences labeled `positive`, `neutral`, `negative`.  
- Used for training and validation.  

---

## Model

- Bi-directional LSTM, 2 layers, embedding_dim=256, hidden_dim=128  
- Outputs 3-class sentiment prediction  
- Trained for 100 epochs using Adam and weighted cross-entropy loss  
- Saved models are in `backend/saved_models/`  

---

## Setup Instructions

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```
###Frontend
```bash
cd frontend
npm install
npm start

```
###Validation
```bash
python validate.py
```
-Shows Accuracy, Precision, Recall & F1 Score.
---
###License
MIT License



