import React, { useState } from 'react';

<<<<<<< HEAD
const SentimentForm = () => {
    const [inputText, setInputText] = useState('');
    const [sentiment, setSentiment] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const response = await fetch('http://localhost:5000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: inputText }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            setSentiment(data.sentiment);
        } catch (err) {
            setError('Error analyzing sentiment');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Enter text for sentiment analysis"
                    required
                />
                <button type="submit" disabled={loading}>
                    {loading ? 'Analyzing...' : 'Analyze Sentiment'}
                </button>
            </form>
            {sentiment && <h3>Sentiment: {sentiment}</h3>}
            {error && <h3 style={{ color: 'red' }}>{error}</h3>}
        </div>
    );
};
=======
function SentimentForm({ setResult, setScore }) {
  const [sentence, setSentence] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult('');
    setScore(null);
    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence }),
      });
      const data = await res.json();
      setResult(data.sentiment);
      setScore(data.score);
    } catch (error) {
      setResult('Error connecting to backend');
      setScore(null);
    }
    setLoading(false);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={sentence}
        onChange={e => setSentence(e.target.value)}
        placeholder="Enter Amharic sentence"
        required
      />
      <button type="submit" disabled={loading}>
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>
    </form>
  );
}
>>>>>>> 1ccb732e (Initial commit from new folder)

export default SentimentForm;