"use client";

import { useState } from 'react';
import styles from './page.module.css';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setFeedbackSent(false);
      setFeedbackMessage(null);
      await processImage(selectedFile);
    }
  };

  const processImage = async (selectedFile: File) => {
    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error('Failed to process image on the server.');
      }

      const data = await res.json();
      if (data.error) {
        throw new Error(data.error);
      }
      
      setResults(data);
    } catch (err: any) {
      setError(err.message || 'An unknown error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async (trueLabel: string) => {
    if (!results) return;
    try {
      const res = await fetch(`${API_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          true_label: trueLabel,
          percepts: results.percepts,
        }),
      });
      const data = await res.json();
      setFeedbackSent(true);
      setFeedbackMessage(data.message || 'Feedback submitted.');
    } catch {
      setFeedbackMessage('Failed to send feedback.');
    }
  };

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <h1 className={styles.title}>🔭 Galaxy Classifier AI</h1>
        <p className={styles.subtitle}>
          Upload a deep space image. Our Multi-Modal Agent will extract visual percepts and deduce the galaxy morphology.
        </p>
      </header>

      <section className={styles.uploadContainer}>
        <span className={styles.uploadIcon}>🌌</span>
        <h2 style={{ marginBottom: '1rem', color: 'white' }}>Select an Image</h2>
        <label className={styles.uploadLabel}>
          Browse Files
          <input 
            type="file" 
            accept="image/jpeg, image/png, image/jpg" 
            onChange={handleFileChange} 
            className={styles.fileInput} 
          />
        </label>
        {file && <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>{file.name}</p>}
        {error && <p style={{ marginTop: '1rem', color: '#ef4444' }}>{error}</p>}
      </section>

      {loading && (
        <div style={{ textAlign: 'center' }}>
          <div className={styles.loader}></div>
          <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Analyzing galaxy morphology...</p>
        </div>
      )}

      {results && (
        <section className={styles.resultsGrid}>
          
          {/* Card 1: Raw Image */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>1. Raw Perception</h3>
            {results.rawBase64 && (
              <img 
                src={`data:image/jpeg;base64,${results.rawBase64}`} 
                alt="Raw Galaxy" 
                className={styles.imagePreview} 
              />
            )}
          </div>

          {/* Card 2: Grad-CAM */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>2. AI Attention (Grad-CAM)</h3>
            {results.heatmapBase64 ? (
              <img 
                src={`data:image/jpeg;base64,${results.heatmapBase64}`} 
                alt="Grad-CAM Heatmap" 
                className={styles.imagePreview} 
              />
            ) : (
              <p style={{color: 'var(--text-secondary)'}}>Heatmap not available</p>
            )}
          </div>

          {/* Card 3: Logical Deduction */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>3. Logical Deduction</h3>
            <div style={{ marginBottom: '1.5rem' }}>
              <p style={{ marginBottom: '0.5rem', fontWeight: 600 }}>Visual Percepts:</p>
              {Object.entries(results.percepts).map(([key, val]: any) => (
                <div key={key} className={styles.progressBarContainer}>
                  <div className={styles.progressLabel}>
                    <span>{key}</span>
                    <span>{(val * 100).toFixed(1)}%</span>
                  </div>
                  <div className={styles.progressTrack}>
                    <div className={styles.progressFill} style={{ width: `${val * 100}%` }}></div>
                  </div>
                </div>
              ))}
            </div>

            <div>
              <p style={{ marginBottom: '0.5rem', fontWeight: 600 }}>Agent Belief State:</p>
              {Object.entries(results.beliefs).map(([key, val]: any) => (
                <div key={key} className={styles.progressLabel}>
                  <span>{key}</span>
                  <span style={{ color: 'var(--accent-color)', fontWeight: 600 }}>{Number(val).toFixed(2)}</span>
                </div>
              ))}
            </div>

            <div className={styles.decisionBox}>
              <p style={{ color: 'white', marginBottom: '0.5rem' }}>Final Output</p>
              <h2 className={styles.decisionText}>{results.decision.toUpperCase()}</h2>
            </div>
          </div>

          {/* Card 4: Expert Feedback (Human-in-the-Loop) */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>4. Expert Feedback</h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.9rem' }}>
              Is the AI wrong? As a domain expert, provide the correct label.
              The agent will adjust its internal rule confidences in real-time.
            </p>

            {!feedbackSent ? (
              <div className={styles.feedbackButtons}>
                <button className={styles.feedbackBtn} onClick={() => sendFeedback('Elliptical')}>
                  🟡 Elliptical
                </button>
                <button className={styles.feedbackBtn} onClick={() => sendFeedback('Spiral')}>
                  🔵 Spiral
                </button>
                <button className={styles.feedbackBtn} onClick={() => sendFeedback('Uncertain')}>
                  🟠 Uncertain
                </button>
              </div>
            ) : (
              <div className={styles.feedbackSuccess}>
                <span>✅</span>
                <p>{feedbackMessage}</p>
              </div>
            )}
          </div>

        </section>
      )}
    </main>
  );
}
