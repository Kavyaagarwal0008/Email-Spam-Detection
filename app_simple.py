from flask import Flask, render_template_string, request, jsonify
import re
import os
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
    <title>MailGuard AI | Email Security Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
*{
    margin:0;
    padding:0;
    box-sizing:border-box;
}

:root{
    --bg:#0b1220;
    --card:#111827;
    --card-light:#1f2937;
    --primary:#06b6d4;
    --primary-dark:#0891b2;
    --success:#10b981;
    --danger:#ef4444;
    --text:#f8fafc;
    --muted:#94a3b8;
    --border:rgba(255,255,255,.08);
}

body{
    font-family:'Inter',sans-serif;
    background:
    radial-gradient(circle at top left,
    rgba(6,182,212,.12),
    transparent 35%),
    radial-gradient(circle at bottom right,
    rgba(16,185,129,.12),
    transparent 40%),
    var(--bg);
    color:var(--text);
    min-height:100vh;
    overflow-x:hidden;
}

/* Animated Background */

.bg-gradient{
    position:fixed;
    inset:0;
    pointer-events:none;
    z-index:-1;

    background-image:
    linear-gradient(rgba(255,255,255,.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.02) 1px, transparent 1px);

    background-size:50px 50px;
}

.bg-gradient::before,
.bg-gradient::after{
    content:"";
    position:absolute;
    width:500px;
    height:500px;
    border-radius:50%;
    filter:blur(120px);
    opacity:.18;
}

.bg-gradient::before{
    background:#06b6d4;
    top:-150px;
    left:-150px;
    animation:blobMove 12s ease-in-out infinite;
}

.bg-gradient::after{
    background:#10b981;
    bottom:-150px;
    right:-150px;
    animation:blobMove 15s ease-in-out infinite reverse;
}

@keyframes blobMove{
    0%,100%{
        transform:translate(0,0) scale(1);
    }
    50%{
        transform:translate(60px,-40px) scale(1.15);
    }
}

/* Page Animation */

.container{
    max-width:1400px;
    margin:auto;
    padding:30px;
    animation:pageLoad .8s ease;
}

@keyframes pageLoad{
    from{
        opacity:0;
        transform:translateY(20px);
    }
    to{
        opacity:1;
        transform:translateY(0);
    }
}

/* Header */

.header{
    display:flex;
    justify-content:space-between;
    align-items:center;
    flex-wrap:wrap;
    gap:20px;
    margin-bottom:35px;
}

.logo-area h1{
    font-size:34px;
    font-weight:800;
    background:
    linear-gradient(
    135deg,
    var(--primary),
    var(--success));
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.logo-area h1:hover{
    filter:drop-shadow(0 0 12px rgba(6,182,212,.5));
}

.logo-area p{
    color:var(--muted);
    margin-top:6px;
}

.model-badge{
    padding:10px 20px;
    border-radius:50px;
    background:rgba(6,182,212,.12);
    border:1px solid rgba(6,182,212,.2);
    backdrop-filter:blur(10px);
}

.model-badge span{
    color:#67e8f9;
    font-weight:600;
}

/* Stats */

.stats-grid{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:20px;
    margin-bottom:30px;
}

/* Cards */

.stat-card,
.input-card,
.history-sidebar{
    position:relative;
    overflow:hidden;

    background:rgba(17,24,39,.8);
    backdrop-filter:blur(16px);

    border:1px solid var(--border);
    border-radius:24px;

    box-shadow:
    0 10px 30px rgba(0,0,0,.25);

    transition:all .4s ease;
}

.stat-card{
    padding:24px;
}

.input-card,
.history-sidebar{
    padding:28px;
}

.stat-card:hover,
.input-card:hover,
.history-sidebar:hover{
    transform:translateY(-6px);
    border-color:rgba(6,182,212,.3);

    box-shadow:
    0 20px 60px rgba(0,0,0,.4),
    0 0 25px rgba(6,182,212,.12);
}

.stat-card::before,
.input-card::before,
.history-sidebar::before{
    content:"";
    position:absolute;
    top:0;
    left:-120%;
    width:60%;
    height:100%;

    background:
    linear-gradient(
    90deg,
    transparent,
    rgba(255,255,255,.08),
    transparent);

    transition:.8s;
}

.stat-card:hover::before,
.input-card:hover::before,
.history-sidebar:hover::before{
    left:150%;
}

.stat-icon{
    width:55px;
    height:55px;
    border-radius:16px;
    background:rgba(6,182,212,.12);

    display:flex;
    align-items:center;
    justify-content:center;

    margin-bottom:15px;
}

.stat-icon i{
    color:var(--primary);
    font-size:24px;
}

.stat-value{
    font-size:34px;
    font-weight:700;
}

.stat-card:hover .stat-value{
    animation:floatText 1s ease;
}

@keyframes floatText{
    50%{
        transform:translateY(-4px);
    }
}

.stat-label{
    color:var(--muted);
    margin-top:5px;
}

/* Main Grid */

.main-grid{
    display:grid;
    grid-template-columns:1fr 380px;
    gap:25px;
}

/* Input */

.input-header h2{
    margin-bottom:8px;
}

.input-header p{
    color:var(--muted);
}

textarea{
    width:100%;
    min-height:220px;

    margin-top:18px;

    background:#0b1220;
    color:white;

    border:1px solid rgba(255,255,255,.08);
    border-radius:18px;

    padding:18px;

    font-size:15px;
    font-family:'Inter',sans-serif;
    line-height:1.7;

    resize:vertical;

    transition:.3s;
}

textarea:focus{
    outline:none;
    transform:scale(1.01);

    border-color:var(--primary);

    box-shadow:
    0 0 0 4px rgba(6,182,212,.15),
    0 10px 30px rgba(6,182,212,.15);
}

textarea::placeholder{
    color:#64748b;
}

/* Buttons */

.button-group{
    display:flex;
    gap:12px;
    margin-top:20px;
}

.btn{
    border:none;
    cursor:pointer;
    border-radius:50px;
    padding:14px 24px;
    font-weight:600;
    transition:.3s;
}

.btn-primary{
    flex:1;
    color:white;

    position:relative;
    overflow:hidden;

    background:
    linear-gradient(
    135deg,
    var(--primary),
    var(--primary-dark));
}

.btn-primary::before{
    content:"";
    position:absolute;
    inset:0;

    background:
    linear-gradient(
    120deg,
    transparent,
    rgba(255,255,255,.35),
    transparent);

    transform:translateX(-100%);
}

.btn-primary:hover::before{
    animation:shine .8s ease;
}

@keyframes shine{
    to{
        transform:translateX(250%);
    }
}

.btn-primary:hover{
    transform:translateY(-3px);
    box-shadow:0 15px 35px rgba(6,182,212,.35);
}

.btn-secondary{
    background:#1e293b;
    color:white;
    border:1px solid rgba(255,255,255,.08);
}

.btn-secondary:hover{
    background:#273549;
}

/* Result */

.result-card{
    margin-top:25px;
    padding:25px;
    border-radius:20px;
    display:none;
    animation:resultAppear .5s ease;
}

@keyframes resultAppear{
    0%{
        opacity:0;
        transform:translateY(25px) scale(.95);
    }
    100%{
        opacity:1;
        transform:translateY(0) scale(1);
    }
}

.result-card.spam{
    background:rgba(239,68,68,.08);
    border:1px solid rgba(239,68,68,.3);
}

.result-card.ham{
    background:rgba(16,185,129,.08);
    border:1px solid rgba(16,185,129,.3);
}

.result-title{
    font-size:22px;
    font-weight:700;
}

.confidence-badge{
    background:#0b1220;
    padding:8px 14px;
    border-radius:30px;
}

.probability-bar{
    height:10px;
    background:#1e293b;
    border-radius:20px;
    overflow:hidden;
    margin-top:20px;
}

.probability-fill{
    height:100%;
    background:
    linear-gradient(
    90deg,
    #06b6d4,
    #10b981);

    position:relative;
}

.probability-fill::after{
    content:"";
    position:absolute;
    inset:0;

    background:
    linear-gradient(
    90deg,
    transparent,
    rgba(255,255,255,.4),
    transparent);

    animation:scanBar 2s linear infinite;
}

@keyframes scanBar{
    from{
        transform:translateX(-100%);
    }
    to{
        transform:translateX(300%);
    }
}

/* History */

.history-header{
    display:flex;
    justify-content:space-between;
    margin-bottom:20px;
    border-bottom:1px solid var(--border);
    padding-bottom:15px;
}

.history-item{
    background:#172033;
    border:1px solid rgba(255,255,255,.05);

    border-radius:14px;
    padding:15px;
    margin-bottom:12px;

    transition:.3s;
}

.history-item:hover{
    transform:translateX(6px);
    border-color:rgba(6,182,212,.3);
}

.history-item.spam{
    border-left:4px solid var(--danger);
}

.history-item.ham{
    border-left:4px solid var(--success);
}

.history-snippet{
    color:#dbeafe;
    line-height:1.6;
    font-size:13px;
}

.history-meta{
    display:flex;
    justify-content:space-between;
    margin-top:10px;
    color:var(--muted);
}

.history-badge{
    padding:4px 10px;
    border-radius:20px;
    font-size:11px;
    font-weight:700;
}

.history-badge.spam{
    background:rgba(239,68,68,.15);
    color:#f87171;
}

.history-badge.ham{
    background:rgba(16,185,129,.15);
    color:#34d399;
}

/* Examples */

.examples-section{
    margin-top:30px;
    padding-top:20px;
    border-top:1px solid var(--border);
}

.examples-grid{
    display:flex;
    flex-wrap:wrap;
    gap:12px;
}

.example-chip{
    padding:10px 16px;
    border-radius:30px;
    background:#172033;
    cursor:pointer;
    transition:.3s;
}

.example-chip:hover{
    transform:translateY(-2px);
    background:rgba(6,182,212,.15);
}

/* Footer */

footer{
    margin-top:60px;
    padding:40px 20px;
    text-align:center;
    border-top:1px solid var(--border);
}

footer p{
    color:var(--muted);
    line-height:1.8;
}

.footer-links{
    display:flex;
    justify-content:center;
    flex-wrap:wrap;
    gap:20px;
    margin-top:15px;
}

.footer-links a{
    color:var(--muted);
    text-decoration:none;
    transition:.3s;
}

.footer-links a:hover{
    color:var(--primary);
    transform:translateY(-3px);
}

/* Scrollbar */

::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-thumb{
    background:
    linear-gradient(
    180deg,
    #06b6d4,
    #10b981);

    border-radius:20px;
}

::-webkit-scrollbar-track{
    background:#111827;
}

/* Responsive */

@media(max-width:968px){

    .main-grid{
        grid-template-columns:1fr;
    }

    .stats-grid{
        grid-template-columns:1fr;
    }
}

@media(max-width:600px){

    .container{
        padding:16px;
    }

    .button-group{
        flex-direction:column;
    }

    .btn{
        width:100%;
    }

    .logo-area h1{
        font-size:28px;
    }
}


.history-header{
    cursor:pointer;
}

.history-arrow{
    transition:all .3s ease;
    color:var(--muted);
}

.history-arrow.rotate{
    transform:rotate(-180deg);
}

#historyWrapper{
    overflow:hidden;
    max-height:600px;
    transition:max-height .4s ease;
}

#historyWrapper.collapsed{
    max-height:0;
}

#historyCountBadge{
    font-size:12px;
    color:var(--primary);
    margin-left:5px;
}

.history-sidebar{
    display:block !important;
    width:100%;
}

#historyWrapper{
    overflow:hidden;
    transition:max-height .4s ease;
}

#historyWrapper.collapsed{
    max-height:0 !important;
}

.history-list{
    max-height:450px;
    overflow-y:auto;
}


#historyListContainer{
    max-height:500px;
    overflow:hidden;
    transition:max-height .4s ease;
}

#historyListContainer.closed{
    max-height:0;
}

#historyArrow{
    transition:.3s;
}

#historyArrow.rotate{
    transform:rotate(-180deg);
}



.clear-btn{
    background:rgba(239,68,68,.12);
    color:#f87171;

    border:1px solid rgba(239,68,68,.25);
    border-radius:12px;

    padding:8px 14px;

    font-size:12px;
    font-weight:600;

    cursor:pointer;

    display:inline-flex;
    align-items:center;
    gap:6px;

    transition:all .3s ease;
}

.clear-btn:hover{
    background:rgba(239,68,68,.22);
    border-color:rgba(239,68,68,.5);

    color:white;

    transform:translateY(-2px);

    box-shadow:
        0 8px 20px rgba(239,68,68,.25),
        0 0 15px rgba(239,68,68,.15);
}

.clear-btn:active{
    transform:scale(.95);
}

.clear-btn i{
    transition:transform .3s ease;
}

.clear-btn:hover i{
    transform:rotate(-12deg);
}


.clear-btn{
    position:relative;
    overflow:hidden;
}

.clear-btn::before{
    content:"";
    position:absolute;
    top:0;
    left:-120%;

    width:60%;
    height:100%;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,.25),
            transparent
        );

    transition:.6s;
}

.clear-btn:hover::before{
    left:150%;
}

</style>
</head>
<body>
    <div class="bg-gradient"></div>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="logo-area">
                <h1><i class="fas fa-shield-alt"></i>MailGuard AI</h1>
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

                <div class="history-header" onclick="toggleHistory()">

                    <div class="history-title">
                        <i class="fas fa-history"></i>
                        Analysis History
                        
                    </div>

                     <div>
                        <button class="clear-btn" onclick="event.stopPropagation();clearHistory();">
                            Clear All
                        </button>

                        <i id="historyArrow" class="fas fa-chevron-down"></i>
                    </div>
                </div>

                <div id="historyListContainer">

                    <div id="historyList" class="history-list">
                        <div class="empty-history">
                            <i class="fas fa-inbox"></i><br>
                            No analyses yet.<br>
                            Start by analyzing an email.
                        </div>
                    </div>

                </div>

            </div>
        </div>

        <footer>
            <h3>Kavya Agarwal</h3>
            <p>AI/ML Developer | Full Stack Developer</p>

            <div class="footer-links">
                <a href="https://mail.google.com/mail/?view=cm&fs=1&to=kavyaagarwal580@gmail.com" target="_blank">
                    <i class="fas fa-envelope"></i> Contact
                </a>

                <a href="https://www.linkedin.com/in/kavya-agarwal-36570b314/" target="_blank">
                    <i class="fab fa-linkedin"></i> LinkedIn
                </a>

                <a href="https://github.com/Kavyaagarwal0008" target="_blank">
                    <i class="fab fa-github"></i> GitHub
                </a>
            </div>

            <p style="margin-top:15px;">
                © 2026 MailGuard AI. Developed by Kavya Agarwal.
            </p>
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

                document.getElementById('emailInput')
                    .scrollIntoView({
                        behavior: 'smooth',
                        block: 'center'
                    });

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
        
        function toggleHistory(){

            const list =document.getElementById("historyListContainer");

            const arrow =document.getElementById("historyArrow");

            list.classList.toggle("closed");
            arrow.classList.toggle("rotate");
        }
        
        // Load history on page load
        loadHistory();
    </script>

    <script src="https://voxifyai.onrender.com/assistant.js" data-user-id="6a23a5b151b38daccb71ba19"> </script>
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
    print(f"🤖 Model accuracy: {accuracy:.2%}")
    print(f"💾 Database: {DATABASE}")
    print("="*50 + "\n")

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
