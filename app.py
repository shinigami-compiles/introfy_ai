import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import docx
import tempfile
from logic import IntroFY

# --- CONFIGURATION ---
app = Flask(__name__)
app.secret_key = "introfy_secret_key"  # Required for session/flash messages
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), "uploads")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size 16MB

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- LAZY-INITIALIZED IntroFY AI SCORER ---
# We lazy-load this so heavy models are only loaded on first /analyze request,
# not during app startup (important for deployment timeouts).
scorer = None

def get_scorer():
    global scorer
    if scorer is None:
        print("... Lazy-loading IntroFY AI Scorer (first /analyze request) ...")
        scorer = IntroFY()
    return scorer

# --- HELPER FUNCTIONS ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'docx'}

def read_file_content(filepath, filename):
    """Reads text from .txt or .docx files"""
    ext = filename.rsplit('.', 1)[1].lower()
    
    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
            
    elif ext == 'docx':
        doc = docx.Document(filepath)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    
    return ""

# --- ROUTES ---

@app.route('/')
def home():
    """Renders the Home Page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles Form Submission and Scoring"""
    transcript_text = ""
    duration = 0
    
    # 1. Get Duration (Required)
    try:
        duration = float(request.form.get('duration', 0))
    except ValueError:
        flash("Invalid duration provided.")
        return redirect(url_for('home'))

    # 2. Get Transcript (File OR Paste)
    if 'file_upload' in request.files and request.files['file_upload'].filename != '':
        file = request.files['file_upload']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                transcript_text = read_file_content(filepath, filename)
            except Exception as e:
                flash(f"Error reading file: {str(e)}")
                return redirect(url_for('home'))
        else:
            flash("Invalid file type. Please upload .txt or .docx")
            return redirect(url_for('home'))
            
    else:
        # Fallback to pasted text
        transcript_text = request.form.get('text_input', '').strip()

    # 3. Validate Input
    if not transcript_text or len(transcript_text) < 10:
        flash("Please provide a valid transcript (at least 10 characters).")
        return redirect(url_for('home'))

    if duration <= 0:
        flash("Please enter a valid speech duration in seconds.")
        return redirect(url_for('home'))

    # 4. Run Analysis using Logic.py (lazy-loaded scorer)
    s = get_scorer()
    results = s.analyze(transcript_text, duration)

    # 5. Render Results Page
    return render_template(
        'result.html', 
        score=results['overall_score'], 
        breakdown=results['breakdown'],
        transcript_snippet=transcript_text[:200] + "..."
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
