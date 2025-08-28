# Amharic Sentiment Analysis Frontend

This project is a frontend application for performing sentiment analysis on Amharic text. It is built using React and communicates with a Python backend that utilizes PyTorch for sentiment classification.

## Project Structure

- **public/index.html**: The main HTML file for the React application.
- **src/App.js**: The main component that sets up the application structure.
- **src/components/SentimentForm.js**: A form component for user input to submit text for sentiment analysis.
- **src/styles/App.css**: CSS styles for the frontend application.
- **package.json**: Configuration file for npm, listing dependencies and scripts.

## Setup Instructions

1. **Clone the Repository**: 
   ```
   git clone <repository-url>
   cd amharic-sentiment-analysis/frontend
   ```

2. **Install Dependencies**: 
   ```
   npm install
   ```

3. **Run the Application**: 
   ```
   npm start
   ```

   This will start the development server and open the application in your default web browser.

## Usage

- Enter Amharic text in the provided form and submit it to receive sentiment analysis results.
- The application will display whether the sentiment of the text is positive, negative, or neutral based on the model's predictions.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License.