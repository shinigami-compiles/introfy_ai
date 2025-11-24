import sys
import subprocess

def install_dependencies():
    print("... Installing Python libraries from requirements.txt ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_nltk_data():
    print("... Downloading NLTK data ...")
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('brown')
    print("NLTK data downloaded successfully.")

def download_spacy_model():
    print("... Verifying Spacy Model ...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("Spacy model 'en_core_web_sm' is ready.")
    except OSError:
        print("Error: Spacy model not found. Please run: python -m spacy download en_core_web_sm")
    except ImportError:
        print("Spacy library not found. Installing dependencies first...")

if __name__ == "__main__":
    print("--- Starting Environment Setup for Flask App ---")
    # 1. Install libraries
    # install_dependencies() # Uncomment if you haven't run pip install yet
    
    # 2. Download NLTK data
    download_nltk_data()
    
    # 3. Verify Spacy
    download_spacy_model()
    
    print("--- Setup Complete. Ready for Backend Logic! ---")