import spacy

# LanguageTool is optional: available on Render/GitHub, may be missing or broken locally
try:
    import language_tool_python
except ImportError:
    language_tool_python = None

import nltk
import re
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize, sent_tokenize


# --- 1. INITIALIZATION & CONFIGURATION ---
class IntroFY:
    def __init__(self):
        print("... Loading AI Models (This may take 10-20 seconds on first run) ...")

        # Load Spacy for NER (Named Entity Recognition)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed: auto-download (works on fresh envs like Render)
            from spacy.cli import download
            print("Spacy model not found. Downloading 'en_core_web_sm'...")
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print("Spacy failed to load, falling back to blank 'en' model:", e)
            self.nlp = spacy.blank("en")

        # --- Grammar: LanguageTool (optional) + safe fallback ---
        self.tool = None
        if language_tool_python is not None:
            try:
                # Try local LanguageTool server (on platforms where it works)
                self.tool = language_tool_python.LanguageTool('en-US')
                print("LanguageTool (local server) initialized.")
            except Exception as e:
                print("Local LanguageTool failed, trying public API:", e)
                try:
                    # Use public HTTP API (works without Java or local server)
                    self.tool = language_tool_python.LanguageToolPublicAPI('en-US')
                    print("LanguageToolPublicAPI initialized.")
                except Exception as e2:
                    print("LanguageTool completely unavailable, using fallback grammar:", e2)
                    self.tool = None
        else:
            print("language_tool_python not installed. Using fallback grammar scoring.")

        # Load VADER for Sentiment
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Load Sentence Transformer for Semantic Similarity
        # 'all-MiniLM-L6-v2' is fast and accurate for this use case
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Define Rubric Constants (Derived from CSV)
        self.filler_words = [
            "um", "uh", "like", "you know", "so", "actually", "basically",
            "right", "i mean", "well", "kinda", "sort of", "okay", "hmm", "ah"
        ]

        # Semantic Anchors: We compare user text against these concept vectors
        self.concepts = {
            "family": ["I live with my family", "My parents", "siblings", "mother and father"],
            "hobbies": ["I like playing", "My hobby is", "I enjoy cricket", "free time", "pastime"],
            "ambition": ["I want to become", "My goal is", "My dream", "future", "ambition", "career"],
            "origin": ["I am from", "I live in", "born in", "native place"]
        }

    # --- 2. CORE SCORING FUNCTIONS ---

    def calculate_speech_rate(self, word_count, duration_sec):
        """Rubric: Ideal 111-140 WPM"""
        if duration_sec <= 0:
            return 0, "Invalid duration (0s)"

        wpm = (word_count / duration_sec) * 60

        # Scoring Logic from CSV
        if 111 <= wpm <= 140:
            score = 10
            feedback = "Perfect pace (Ideal)."
        elif 141 <= wpm <= 160:
            score = 6
            feedback = "Slightly fast."
        elif 81 <= wpm <= 110:
            score = 6
            feedback = "Slightly slow."
        elif wpm > 161:
            score = 2
            feedback = "Too fast."
        else:  # < 80
            score = 2
            feedback = "Too slow."

        return score, f"{int(wpm)} WPM - {feedback}"

    def calculate_grammar(self, text, word_count):
        """
        Grammar scoring.

        If LanguageTool is available (Render / cloud), use it.
        Otherwise, fall back to a heuristic grammar score so local runs don't break.
        """
        if word_count == 0:
            return 0, "Empty text provided."

        # --- Path 1: Use LanguageTool when available ---
        if self.tool is not None:
            try:
                matches = self.tool.check(text)
                error_count = len(matches)

                errors_per_100 = (error_count * 100) / word_count
                metric = 1 - min(errors_per_100 / 10, 1)

                if metric > 0.9:
                    score = 10
                elif 0.7 <= metric <= 0.89:
                    score = 8
                elif 0.5 <= metric <= 0.69:
                    score = 6
                elif 0.3 <= metric <= 0.49:
                    score = 4
                else:
                    score = 2

                return score, f"{error_count} grammar issues detected. (LanguageTool score: {metric:.2f})"
            except Exception as e:
                # If LT fails at runtime (network, API, etc.), fall through to heuristic fallback
                print("LanguageTool error during grammar check, using fallback:", e)

        # --- Path 2: Fallback heuristic (no LanguageTool) ---
        # Simple rules: sentence-end punctuation, capitalisation, 'i' vs 'I'
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        bad_caps = 0
        for s in sentences:
            if s and s[0].islower():
                bad_caps += 1

        lower_i = text.count(" i ")
        long_sentences = sum(1 for s in sentences if len(s.split()) > 25)

        heuristic_errors = bad_caps + lower_i + long_sentences
        errors_per_100 = (heuristic_errors * 100) / word_count
        metric = max(0.0, 1 - min(errors_per_100 / 10, 1))

        if metric > 0.9:
            score = 9
        elif 0.7 <= metric <= 0.89:
            score = 7
        elif 0.5 <= metric <= 0.69:
            score = 5
        elif 0.3 <= metric <= 0.49:
            score = 3
        else:
            score = 2

        fb_parts = [f"Heuristic grammar score: {metric:.2f}"]
        if bad_caps:
            fb_parts.append(f"{bad_caps} sentence(s) start without capital letter.")
        if lower_i:
            fb_parts.append(f"Found {lower_i} lowercase 'i' used as pronoun.")
        if long_sentences:
            fb_parts.append(f"{long_sentences} very long sentence(s) detected (>25 words).")

        feedback = " ".join(fb_parts) or "Basic grammar looks okay (fallback heuristic)."
        return score, feedback

    def calculate_vocabulary(self, text, word_count):
        """Rubric: TTR = Distinct words / Total words"""
        tokens = word_tokenize(text.lower())
        # Filter out punctuation to get pure words
        words = [word for word in tokens if word.isalpha()]

        if not words:
            return 0, "No valid words"

        unique_words = set(words)
        ttr = len(unique_words) / len(words)

        # CSV Thresholds
        if 0.9 <= ttr <= 1.0:
            score = 10
        elif 0.7 <= ttr <= 0.89:
            score = 8
        elif 0.5 <= ttr <= 0.69:
            score = 6
        elif 0.3 <= ttr <= 0.49:
            score = 4
        else:
            score = 2

        return score, f"TTR: {ttr:.2f} (Unique: {len(unique_words)}/{len(words)})"

    def calculate_clarity(self, text, word_count):
        """Rubric: Filler Word Rate"""
        tokens = word_tokenize(text.lower())
        filler_count = sum(1 for word in tokens if word in self.filler_words)

        if word_count == 0:
            return 0, "No text"

        rate = (filler_count / word_count) * 100

        # CSV Thresholds
        if 0 <= rate <= 3:
            score = 15
        elif 4 <= rate <= 6:
            score = 12
        elif 7 <= rate <= 9:
            score = 9
        elif 10 <= rate <= 12:
            score = 6
        else:
            score = 3

        return score, f"Filler Rate: {rate:.1f}% ({filler_count} fillers found)"

    def calculate_engagement(self, text):
        """Rubric: Sentiment/Positivity probability"""
        scores = self.sentiment_analyzer.polarity_scores(text)
        metric = scores['pos']
        compound = scores['compound']

        # If text is highly positive overall, treat it as high engagement
        final_metric = max(metric, compound)

        # Logic Table
        if final_metric >= 0.9:
            score = 15
        elif 0.7 <= final_metric <= 0.89:
            score = 12
        elif 0.5 <= final_metric <= 0.69:
            score = 9
        elif 0.3 <= final_metric <= 0.49:
            score = 6
        else:
            score = 3

        return score, f"Positivity Index: {final_metric:.2f}"

    def check_content_structure(self, text):
        """
        Rubric Analysis:
        1. Salutation (5pts)
        2. Key Entities (Name, Age, School, Family, Hobbies) (30pts)
        3. Flow (5pts)
        """
        doc = self.nlp(text)
        text_lower = text.lower()

        # A. SALUTATION (5 Marks)
        salutations = ["good morning", "good afternoon", "good evening", "hello", "hi ", "hey "]
        has_salutation = any(s in text_lower[:60] for s in salutations)
        score_salutation = 5 if has_salutation else 0

        # B. KEYWORDS & ENTITIES (30 Marks Total)
        detected = []
        score_content = 0

        # 1. Name (Must Have - 4pts)
        has_name_kw = "my name" in text_lower or "myself" in text_lower or "i am" in text_lower
        has_person_ent = any(ent.label_ == "PERSON" for ent in doc.ents)
        if has_name_kw and has_person_ent:
            score_content += 4
            detected.append("Name")

        # 2. Age (Must Have - 4pts)
        if re.search(r'\b(years old|age|born in)\b', text_lower):
            score_content += 4
            detected.append("Age")

        # 3. School/Class (Must Have - 4pts)
        if "school" in text_lower or "class" in text_lower or "student" in text_lower:
            score_content += 4
            detected.append("School/Class")

        # 4. Family (Must Have - 4pts)
        if self.check_semantic_similarity(text, self.concepts["family"]):
            score_content += 4
            detected.append("Family")

        # 5. Hobbies (Must Have - 4pts)
        if self.check_semantic_similarity(text, self.concepts["hobbies"]):
            score_content += 4
            detected.append("Hobbies")

        # 6. Origin (Good to have - 2pts)
        if "from" in text_lower or any(ent.label_ == "GPE" for ent in doc.ents):
            score_content += 2
            detected.append("Origin")

        # 7. Ambition (Good to have - 2pts)
        if self.check_semantic_similarity(text, self.concepts["ambition"]):
            score_content += 2
            detected.append("Ambition")

        # 8. Unique/Fun Fact (Good to have - 4pts from remaining pool)
        if "fun fact" in text_lower or "unique" in text_lower or "special" in text_lower:
            score_content += 4
            detected.append("Unique Fact")

        # Cap Content Score at 30
        score_content = min(score_content, 30)

        # C. FLOW (5 Marks)
        has_closing = "thank" in text_lower[-150:]  # Check last portion

        score_flow = 0
        if has_salutation and has_closing:
            score_flow = 5
        elif has_salutation or has_closing:
            score_flow = 3

        return {
            "salutation_score": score_salutation,
            "content_score": score_content,
            "flow_score": score_flow,
            "detected_keywords": detected
        }

    def check_semantic_similarity(self, text, anchor_phrases):
        """Returns True if text is semantically similar to any anchor phrase"""
        doc_sentences = sent_tokenize(text)
        if not doc_sentences:
            return False

        # Encode all sentences in the transcript
        doc_embeddings = self.semantic_model.encode(doc_sentences, convert_to_tensor=True)
        # Encode anchor phrases (e.g., "I like playing cricket")
        anchor_embeddings = self.semantic_model.encode(anchor_phrases, convert_to_tensor=True)

        cosine_scores = util.cos_sim(doc_embeddings, anchor_embeddings)

        max_score = float(cosine_scores.max())
        return max_score > 0.35

    # --- 3. MAIN PIPELINE ---
    def analyze(self, transcript_text, duration_sec):
        # Basic stats
        word_count = len(word_tokenize(transcript_text))

        # 1. Speech Rate (10%)
        s_rate, fb_rate = self.calculate_speech_rate(word_count, duration_sec)

        # 2. Grammar (10%)
        s_gram, fb_gram = self.calculate_grammar(transcript_text, word_count)

        # 3. Vocabulary (10%)
        s_vocab, fb_vocab = self.calculate_vocabulary(transcript_text, word_count)

        # 4. Clarity (15%)
        s_clar, fb_clar = self.calculate_clarity(transcript_text, word_count)

        # 5. Engagement (15%)
        s_eng, fb_eng = self.calculate_engagement(transcript_text)

        # 6. Content & Structure (40%)
        structure_data = self.check_content_structure(transcript_text)
        s_salut = structure_data['salutation_score']
        s_cont = structure_data['content_score']
        s_flow = structure_data['flow_score']

        # TOTAL SCORE CALCULATION
        total_score = s_rate + s_gram + s_vocab + s_clar + s_eng + s_salut + s_cont + s_flow
        total_score = min(total_score, 100)  # Cap at 100 just in case

        return {
            "overall_score": round(total_score, 1),
            "breakdown": [
                {"criteria": "Speech Rate", "score": s_rate, "max": 10, "feedback": fb_rate},
                {"criteria": "Grammar", "score": s_gram, "max": 10, "feedback": fb_gram},
                {"criteria": "Vocabulary", "score": s_vocab, "max": 10, "feedback": fb_vocab},
                {"criteria": "Clarity (Fillers)", "score": s_clar, "max": 15, "feedback": fb_clar},
                {"criteria": "Engagement", "score": s_eng, "max": 15, "feedback": fb_eng},
                {"criteria": "Salutation", "score": s_salut, "max": 5, "feedback": "Presence Checked"},
                {"criteria": "Content Relevance", "score": s_cont, "max": 30,
                 "feedback": f"Found: {', '.join(structure_data['detected_keywords'])}"},
                {"criteria": "Flow & Structure", "score": s_flow, "max": 5, "feedback": "Checked Order"}
            ]
        }
