# Amharic Sentiment Analysis Project
This project implements a sentiment analysis application for Amharic text using a PyTorch-based backend and a React frontend. The application allows users to input Amharic text and receive sentiment predictions.

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
## Backend

The backend is built using Flask and PyTorch. It includes:

- **app.py**: The main entry point for the backend application, setting up routes for sentiment analysis.
- **model/sentiment_model.py**: The implementation of the sentiment analysis model.
- **utils/preprocess.py**: Utility functions for preprocessing text data.
- **requirements.txt**: Lists the required Python dependencies.

## Frontend

The frontend is built using React. It includes:

- **public/index.html**: The main HTML file for the React application.
- **src/App.js**: The main component of the React application.
- **src/components/SentimentForm.js**: A form component for user input.
- **src/styles/App.css**: CSS styles for the frontend application.
- **package.json**: Configuration file for npm.

>>>>>>> 1ccb732e (Initial commit from new folder)
## Setup Instructions

### Backend

1. Navigate to the `backend` directory.
2. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```
3. Run the backend application:
   ```
   python app.py
   ```

### Frontend

1. Navigate to the `frontend` directory.
2. Install the required dependencies using:
   ```
   npm install
   ```
3. Start the frontend application:
   ```
   npm start
   ```

## Usage

Once both the backend and frontend are running, you can access the application in your web browser. Input Amharic text into the form and submit it to receive sentiment predictions.

