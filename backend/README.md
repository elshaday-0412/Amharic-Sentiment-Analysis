# Amharic Sentiment Analysis Backend

This document provides an overview of the backend components of the Amharic Sentiment Analysis project. The backend is built using Flask and PyTorch to handle sentiment classification of Amharic text.

## Project Structure

The backend consists of the following files:

- **app.py**: The main entry point for the backend application. It initializes the Flask app, sets up routes for sentiment analysis, and handles incoming requests.
- **model/sentiment_model.py**: Contains the implementation of the sentiment analysis model using PyTorch. It defines the architecture of the neural network and includes methods for training and inference.
- **utils/preprocess.py**: Includes utility functions for preprocessing text data, such as tokenization, normalization, and converting text to tensor format suitable for the model.
- **requirements.txt**: Lists the Python dependencies required for the backend, including Flask, PyTorch, and any other necessary libraries.

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd amharic-sentiment-analysis/backend
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```
   python app.py
   ```

The backend will start running on `http://localhost:5000`.

## API Endpoints

- **POST /analyze**: Accepts a JSON payload with the text to analyze and returns the sentiment classification.

### Example Request
```
POST /analyze
Content-Type: application/json

{
    "text": "እንዴት ነህ?"
}
```

### Example Response
```
{
    "sentiment": "positive"
}
```

## Usage

Once the backend is running, you can interact with it using tools like Postman or through the frontend application. The frontend will provide a user-friendly interface for submitting text and receiving sentiment analysis results.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.