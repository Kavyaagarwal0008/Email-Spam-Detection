from flask import Flask, render_template_string, request, jsonify
import re
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sqlite3
from datetime import datetime
from contextlib import contextmanager

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('stopwords')

app = Flask(__name__)

# Database setup for history
DATABASE = 'spam_history.db'

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                email_snippet TEXT NOT NULL,
                result_type TEXT NOT NULL,
                spam_probability REAL NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✅ Database initialized")

# Load your trained Kaggle model
print("\n" + "="*50)
print("Loading your trained spam model...")
print("="*50)

try:
    # Load the model you trained with Kaggle dataset
    model_data = joblib.load('spam_model.pkl')
    
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    accuracy = model_data['accuracy']
    training_samples = model_data.get('training_samples', 'N/A')
    
    print(f"✅ Model loaded successfully!")
    print(f"   Model type: {model_data.get('model_type', 'Logistic Regression')}")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   Trained on: {training_samples} messages")
    
except FileNotFoundError:
    print("❌ Error: spam_model.pkl not found!")
    print("   Please make sure the trained model is in the same folder")
    print("   You can train it using the Kaggle dataset in Colab")
    exit(1)

# Preprocessing function (must match training)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    """Clean and preprocess text - matches training preprocessing"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words[:500])

# Initialize database
init_db()

# Professional HTML Template (Updated Frontend Only - Model Code Unchanged)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
    <title>Sentinel | AI Email Security Platform</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: #0a0e1a;
            min-height: 100vh;
            position: relative;
        }

        /* Animated Background */
        .bg-gradient {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
                        radial-gradient(circle at 80% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 50%);
            pointer-events: none;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
            position: relative;
            z-index: 1;
        }

        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 32px;
            flex-wrap: wrap;
            gap: 20px;
        }

        .logo-area h1 {
            font-size: 28px;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .logo-area p {
            font-size: 14px;
            color: #6b7280;
            margin-top: 4px;
        }

        .model-badge {
            background: rgba(99, 102, 241, 0.1);
            backdrop-filter: blur(10px);
            padding: 8px 20px;
            border-radius: 40px;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }

        .model-badge span {
            font-size: 13px;
            color: #8b5cf6;
            font-weight: 500;
        }

        /* Stats Cards */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 32px;
        }

        @media (max-width: 768px) {
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 24px;
            padding: 20px;
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            border-color: rgba(99, 102, 241, 0.3);
            transform: translateY(-2px);
        }

        .stat-icon {
            width: 48px;
            height: 48px;
            background: rgba(99, 102, 241, 0.1);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 16px;
        }

        .stat-icon i {
            font-size: 24px;
            color: #6366f1;
        }

        .stat-value {
            font-size: 32px;
            font-weight: 700;
            color: white;
            margin-bottom: 4px;
        }

        .stat-label {
            font-size: 14px;
            color: #9ca3af;
        }

        /* Main Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 24px;
        }

        @media (max-width: 968px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Input Card */
        .input-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 24px;
            padding: 28px;
        }

        .input-header h2 {
            color: white;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .input-header p {
            color: #9ca3af;
            font-size: 13px;
            margin-bottom: 20px;
        }

        textarea {
            width: 100%;
            padding: 16px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            color: white;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            resize: vertical;
            transition: all 0.3s;
        }

        textarea:focus {
            outline: none;
            border-color: #6366f1;
            background: rgba(0, 0, 0, 0.5);
        }

        textarea::placeholder {
            color: #4b5563;
        }

        .button-group {
            display: flex;
            gap: 12px;
            margin-top: 20px;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 40px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            flex: 1;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(99, 102, 241, 0.3);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: #e5e7eb;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        /* Result Card */
        .result-card {
            margin-top: 24px;
            padding: 24px;
            border-radius: 20px;
            display: none;
            animation: fadeInUp 0.4s ease;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .result-card.spam {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.1) 100%);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .result-card.ham {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.1) 100%);
            border: 1px solid rgba(34, 197, 94, 0.3);
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 20px;
        }

        .result-title {
            font-size: 20px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .confidence-badge {
            background: rgba(0, 0, 0, 0.5);
            padding: 6px 14px;
            border-radius: 40px;
            font-size: 12px;
            font-weight: 500;
            color: #e5e7eb;
        }

        .probability-bar {
            background: rgba(0, 0, 0, 0.3);
            height: 8px;
            border-radius: 4px;
            margin: 20px 0;
            overflow: hidden;
        }

        .probability-fill {
            height: 100%;
            background: linear-gradient(90deg, #f97316, #ef4444);
            transition: width 0.5s ease;
            border-radius: 4px;
        }

        /* History Sidebar */
        .history-sidebar {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 24px;
            padding: 24px;
            height: fit-content;
        }

        .history-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .history-title {
            font-weight: 600;
            color: white;
            font-size: 16px;
        }

        .clear-btn {
            background: none;
            border: none;
            color: #9ca3af;
            cursor: pointer;
            font-size: 12px;
            padding: 6px 12px;
            border-radius: 8px;
            transition: all 0.3s;
        }

        .clear-btn:hover {
            color: #ef4444;
            background: rgba(239, 68, 68, 0.1);
        }

        .history-list {
            max-height: 450px;
            overflow-y: auto;
        }

        .history-item {
            background: rgba(255, 255, 255, 0.03);
            padding: 14px;
            border-radius: 14px;
            margin-bottom: 12px;
            cursor: pointer;
            transition: all 0.3s;
            border-left: 3px solid transparent;
        }

        .history-item:hover {
            background: rgba(255, 255, 255, 0.08);
            transform: translateX(4px);
        }

        .history-item.spam {
            border-left-color: #ef4444;
        }

        .history-item.ham {
            border-left-color: #22c55e;
        }

        .history-snippet {
            font-size: 12px;
            color: #d1d5db;
            margin-bottom: 8px;
            line-height: 1.5;
        }

        .history-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            color: #6b7280;
        }

        .history-badge {
            padding: 2px 8px;
            border-radius: 20px;
            font-size: 10px;
            font-weight: 600;
        }

        .history-badge.spam {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
        }

        .history-badge.ham {
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
        }

        .empty-history {
            text-align: center;
            padding: 40px 20px;
            color: #6b7280;
            font-size: 13px;
        }

        /* Examples Section */
        .examples-section {
            margin-top: 24px;
            padding-top: 24px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .examples-section h3 {
            color: white;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 12px;
        }

        .examples-grid {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }

        .example-chip {
            background: rgba(255, 255, 255, 0.05);
            padding: 8px 16px;
            border-radius: 40px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 12px;
            color: #d1d5db;
        }

        .example-chip:hover {
            background: rgba(99, 102, 241, 0.2);
            transform: translateY(-2px);
        }

        /* Loading Animation */
        .btn-primary.loading {
            position: relative;
            color: transparent;
        }

        .btn-primary.loading::after {
            content: "⏳";
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            from { transform: translate(-50%, -50%) rotate(0deg); }
            to { transform: translate(-50%, -50%) rotate(360deg); }
        }

        footer {
            margin-top: 32px;
            text-align: center;
            padding: 24px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }

        footer p {
            font-size: 12px;
            color: #6b7280;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(99, 102, 241, 0.5);
            border-radius: 3px;
        }

        /* Responsive */
        @media (max-width: 600px) {
            .container {
                padding: 16px;
            }
            .input-card {
                padding: 20px;
            }
            .btn {
                padding: 10px 16px;
            }
        }
    </style>
</head>
<body>
    <div class="bg-gradient"></div>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="logo-area">
                <h1><i class="fas fa-shield-alt"></i> Sentinel</h1>
                <p>AI-Powered Email Security Intelligence Platform</p>
            </div>
            <div class="model-badge">
                <span><i class="fas fa-brain"></i> Logistic Regression • {{ "%.1f"|format(accuracy*100) }}% Validation Accuracy</span>
            </div>
        </div>

        <!-- Stats Cards -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-database"></i></div>
                <div class="stat-value">{{ training_samples }}</div>
                <div class="stat-label">Training Samples Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-chart-line"></i></div>
                <div class="stat-value">{{ "%.1f"|format(accuracy*100) }}<span style="font-size: 18px;">%</span></div>
                <div class="stat-label">Model Validation Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-clock"></i></div>
                <div class="stat-value" id="historyCount">0</div>
                <div class="stat-label">Total Classifications</div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="main-grid">
            <div>
                <div class="input-card">
                    <div class="input-header">
                        <h2><i class="fas fa-envelope"></i> Email Analysis</h2>
                        <p>Paste the email content below for real-time threat detection</p>
                    </div>
                    <textarea id="emailInput" rows="6" placeholder="Paste email content here..."></textarea>
                    <div class="button-group">
                        <button class="btn btn-primary" onclick="classifyEmail()">
                            <i class="fas fa-robot"></i> Analyze
                        </button>
                        <button class="btn btn-secondary" onclick="clearInput()">
                            <i class="fas fa-eraser"></i> Clear
                        </button>
                    </div>
                    <div id="resultCard" class="result-card"></div>
                    
                    <div class="examples-section">
                        <h3><i class="fas fa-lightbulb"></i> Test Samples</h3>
                        <div class="examples-grid">
                            <div class="example-chip" onclick="loadExample('spam')">🔴 Spam Detection</div>
                            <div class="example-chip" onclick="loadExample('ham')">🟢 Legitimate Email</div>
                            <div class="example-chip" onclick="loadExample('phishing')">⚠️ Phishing Attempt</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="history-sidebar">
                <div class="history-header">
                    <div class="history-title"><i class="fas fa-history"></i> Analysis History</div>
                    <button class="clear-btn" onclick="clearHistory()">
                        <i class="fas fa-trash-alt"></i> Clear All
                    </button>
                </div>
                <div id="historyList" class="history-list">
                    <div class="empty-history">
                        <i class="fas fa-inbox"></i><br>
                        No analyses yet.<br>
                        Start by analyzing an email.
                    </div>
                </div>
            </div>
        </div>

        <footer>
            <p>Powered by Machine Learning • Logistic Regression • Trained on {{ training_samples }} labeled messages • Real-time Threat Intelligence</p>
        </footer>
    </div>

    <script>
        async function classifyEmail() {
            const email = document.getElementById('emailInput').value.trim();
            const btn = document.querySelector('.btn-primary');
            
            if (!email) {
                alert('Please enter email content to analyze');
                return;
            }
            
            btn.classList.add('loading');
            btn.disabled = true;
            
            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email: email })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResult(data);
                    loadHistory();
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Connection error. Ensure server is running.');
            } finally {
                btn.classList.remove('loading');
                btn.disabled = false;
            }
        }
        
        function displayResult(data) {
            const card = document.getElementById('resultCard');
            const spamProb = (data.spam_probability * 100).toFixed(1);
            const confidence = (data.confidence * 100).toFixed(1);
            
            card.className = `result-card ${data.result_type}`;
            
            card.innerHTML = `
                <div class="result-header">
                    <div class="result-title">
                        ${data.is_spam ? '<i class="fas fa-exclamation-triangle"></i> Spam Detected' : '<i class="fas fa-check-circle"></i> Safe'}
                    </div>
                    <div class="confidence-badge">Confidence: ${confidence}%</div>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${spamProb}%;"></div>
                </div>
                <div style="margin-top: 16px; display: flex; justify-content: space-between; font-size: 13px;">
                    <span style="color: #f87171;">Spam Probability: ${spamProb}%</span>
                    <span style="color: #4ade80;">Safe Probability: ${(data.ham_probability * 100).toFixed(1)}%</span>
                </div>
            `;
            
            card.style.display = 'block';
            card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        
        async function loadHistory() {
            try {
                const response = await fetch('/api/history');
                const history = await response.json();
                const historyList = document.getElementById('historyList');
                document.getElementById('historyCount').textContent = history.length;
                
                if (history.length === 0) {
                    historyList.innerHTML = '<div class="empty-history"><i class="fas fa-inbox"></i><br>No analyses yet.<br>Start by analyzing an email.</div>';
                    return;
                }
                
                historyList.innerHTML = history.map(item => `
                    <div class="history-item ${item.result_type}" onclick="loadHistoryEmail(${item.id})">
                        <div class="history-snippet">${escapeHtml(item.email_snippet)}</div>
                        <div class="history-meta">
                            <span class="history-badge ${item.result_type}">${item.is_spam ? 'SPAM' : 'SAFE'}</span>
                            <span><i class="far fa-clock"></i> ${item.created_at}</span>
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading history:', error);
            }
        }
        
        async function loadHistoryEmail(id) {
            try {
                const response = await fetch(`/api/history/${id}`);
                const data = await response.json();
                document.getElementById('emailInput').value = data.email;
                classifyEmail();
            } catch (error) {
                console.error('Error loading history:', error);
            }
        }
        
        async function clearHistory() {
            if (!confirm('Clear all analysis history? This action cannot be undone.')) return;
            
            try {
                await fetch('/api/history/clear', { method: 'DELETE' });
                loadHistory();
            } catch (error) {
                console.error('Error clearing history:', error);
            }
        }
        
        function clearInput() {
            document.getElementById('emailInput').value = '';
            document.getElementById('resultCard').style.display = 'none';
        }
        
        function loadExample(type) {
            let example = '';
            if (type === 'spam') {
                example = "FREE! You've won a $1000 gift card! Click here to claim your prize now!";
            } else if (type === 'ham') {
                example = "Hi team, meeting scheduled for tomorrow at 10am. Please confirm your attendance.";
            } else if (type === 'phishing') {
                example = "URGENT: Your PayPal account has been limited! Verify now at http://fake-paypal.com";
            }
            document.getElementById('emailInput').value = example;
            classifyEmail();
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Load history on page load
        loadHistory();
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, accuracy=accuracy, training_samples=training_samples)

@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify email using trained model"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        
        if not email:
            return jsonify({'error': 'Empty email'}), 400
        
        # Predict using your trained model
        processed = preprocess(email)
        features = vectorizer.transform([processed])
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        result = {
            'is_spam': bool(prediction),
            'result_type': 'spam' if prediction == 1 else 'ham',
            'spam_probability': float(probability[1]),
            'ham_probability': float(probability[0]),
            'confidence': float(max(probability))
        }
        
        # Save to database
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO classifications (email, email_snippet, result_type, spam_probability, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                email[:2000],
                email[:100] + ('...' if len(email) > 100 else ''),
                result['result_type'],
                result['spam_probability'],
                result['confidence']
            ))
            result['id'] = cursor.lastrowid
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get classification history"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, email_snippet, result_type, confidence, 
                       datetime(created_at, 'localtime') as created_at
                FROM classifications 
                ORDER BY created_at DESC 
                LIMIT 20
            ''')
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    'id': row['id'],
                    'email_snippet': row['email_snippet'],
                    'result_type': row['result_type'],
                    'is_spam': row['result_type'] == 'spam',
                    'confidence': row['confidence'],
                    'created_at': row['created_at']
                })
            
            return jsonify(history)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<int:id>', methods=['GET'])
def get_history_item(id):
    """Get specific history item"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT email FROM classifications WHERE id = ?', (id,))
            row = cursor.fetchone()
            
            if row:
                return jsonify({'email': row['email']})
            else:
                return jsonify({'error': 'Not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/clear', methods=['DELETE'])
def clear_history():
    """Clear all history"""
    try:
        with get_db() as conn:
            conn.execute('DELETE FROM classifications')
        return jsonify({'message': 'History cleared'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_accuracy': accuracy,
        'model_trained': True
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Spam Classifier with Trained Model")
    print("="*50)
    print(f"📧 Open: http://localhost:5000")
    print(f"🤖 Model accuracy: {accuracy:.2%}")
    print(f"💾 Database: {DATABASE}")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)