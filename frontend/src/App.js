import React, { useState } from 'react';
<<<<<<< HEAD
import './styles/App.css';
import SentimentForm from './components/SentimentForm';

function App() {
  const [result, setResult] = useState(null);

  const handleResult = (sentiment) => {
    setResult(sentiment);
  };
=======
import SentimentForm from './components/SentimentForm';
import './styles/App.css';

function App() {
  const [result, setResult] = useState('');
  const [score, setScore] = useState(null);
>>>>>>> 1ccb732e (Initial commit from new folder)

  return (
    <div className="App">
      <h1>Amharic Sentiment Analysis</h1>
<<<<<<< HEAD
      <SentimentForm onResult={handleResult} />
      {result && <div className="result">Sentiment: {result}</div>}
=======
      <SentimentForm setResult={setResult} setScore={setScore} />
      {result && (
        <div>
          <p>Sentiment: <strong>{result}</strong></p>
          {score !== null && <p>Confidence: {score.toFixed(2)}</p>}
        </div>
      )}
>>>>>>> 1ccb732e (Initial commit from new folder)
    </div>
  );
}

export default App;