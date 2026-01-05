import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox  # Import for the alert popup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- AUTOMATIC DATA DOWNLOADER ---

# This section ensures all required NLP resources are installed on your PC
def setup_nltk():
    resources = ['punkt', 'stopwords', 'punkt_tab', 'averaged_perceptron_tagger']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if 'punkt' in res else f'corpora/{res}')
        except LookupError:
            print(f"Downloading required resource: {res}")
            nltk.download(res)

setup_nltk()

# ---------------------------------------------------------
# ML ENGINE: NLP & Feature Engineering
# ---------------------------------------------------------
class EssayScorerEngine:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Reference topic for "Content Relevance"
        self.prompt_topic = "technology impact on education society learning development digital schools"
        self.vectorizer = TfidfVectorizer()

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        filtered = [w for w in tokens if w.isalnum() and w not in self.stop_words]
        return " ".join(filtered) if filtered else "empty"

    def evaluate(self, text):
        words = text.split()
        word_count = len(words)
        # The line below was causing your error; setup_nltk() now fixes it
        sentences = sent_tokenize(text)
        
        # 1. Content Relevance (TF-IDF Similarity)
        clean_text = self.preprocess(text)
        clean_prompt = self.preprocess(self.prompt_topic)
        tfidf_matrix = self.vectorizer.fit_transform([clean_prompt, clean_text])
        sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        relevance = min(10, (sim_score * 20) + 3) if word_count > 5 else 1.0
        
        # 2. Grammar & Spelling (Check capitalization and basic structure)
        cap_errors = sum(1 for s in sentences if s and not s[0].isupper())
        grammar_base = 10 - (cap_errors * 1.5)
        grammar_score = max(1.0, min(10, grammar_base))

        # 3. Coherence & Structure
        avg_sent_len = word_count / len(sentences) if sentences else 0
        if word_count < 20:
            structure = 2.0
        elif 10 < avg_sent_len < 25:
            structure = 9.0
        else:
            structure = 6.0

        # 4. Vocabulary Richness
        unique_words = len(set(w.lower() for w in words))
        diversity = unique_words / (word_count + 1)
        vocab_score = min(10, (diversity * 15) + (word_count / 150))

        return {
            "Relevance": round(relevance, 1),
            "Grammar": round(grammar_score, 1),
            "Structure": round(structure, 1),
            "Vocabulary": round(vocab_score, 1)
        }

# ---------------------------------------------------------
# GUI: Detailed Score Display
# ---------------------------------------------------------
class EssayScorerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.scorer = EssayScorerEngine()

        self.title("ML Model Automated Essay Scoring System Using NLP")
        self.geometry("900x750")
        ctk.set_appearance_mode("dark")

        self.grid_columnconfigure(0, weight=1)
        
        self.header = ctk.CTkLabel(self, text="Write-Right AI Analysis ", font=("Arial", 28, "bold"))
        self.header.pack(pady=20)

        self.textbox = ctk.CTkTextbox(self, width=800, height=300, font=("Arial", 14))
        self.textbox.pack(pady=10, padx=20)
        self.textbox.insert("0.0", "Paste your essay here...")

        self.analyze_btn = ctk.CTkButton(self, text="Analyze Essay", command=self.process_essay, 
                                         height=45, font=("Arial", 16, "bold"), fg_color="#246abf")
        self.analyze_btn.pack(pady=20)

        self.stats_frame = ctk.CTkFrame(self, fg_color="#202020", corner_radius=12)
        self.stats_frame.pack(pady=10, padx=40, fill="x")

        self.res_relevance = self.create_metric_row("Content Relevance:")
        self.res_grammar = self.create_metric_row("Grammar & Spelling:")
        self.res_structure = self.create_metric_row("Coherence & Structure:")
        self.res_vocab = self.create_metric_row("Vocabulary Richness:")
        
        self.final_score = ctk.CTkLabel(self, text="Total Score: --", font=("Arial", 32, "bold"))
        self.final_score.pack(pady=30)

    def create_metric_row(self, label_text):
        row = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        row.pack(fill="x", padx=30, pady=12)
        lbl = ctk.CTkLabel(row, text=label_text, font=("Arial", 16), width=250, anchor="w")
        lbl.pack(side="left")
        val = ctk.CTkLabel(row, text="0.0 / 10", font=("Arial", 16, "bold"), text_color="#3498db")
        val.pack(side="right")
        return val

    def process_essay(self):
        # 1. Get text and strip white space
        text = self.textbox.get("1.0", "end-1c").strip()
        
        # 2. Check if empty
        if not text or text == "Paste your essay here...":
            # Show Alert
            messagebox.showwarning("Input Error", "Please input some text or words.")
            
            # Reset/Remove previous analysis visuals
            self.res_relevance.configure(text="0.0 / 10")
            self.res_grammar.configure(text="0.0 / 10")
            self.res_structure.configure(text="0.0 / 10")
            self.res_vocab.configure(text="0.0 / 10")
            self.final_score.configure(text="Total Score: --", text_color="white")
            return

        # 3. Proceed with calculation if text is present
        results = self.scorer.evaluate(text)
        
        self.res_relevance.configure(text=f"{results['Relevance']} / 10")
        self.res_grammar.configure(text=f"{results['Grammar']} / 10")
        self.res_structure.configure(text=f"{results['Structure']} / 10")
        self.res_vocab.configure(text=f"{results['Vocabulary']} / 10")
        
        total = sum(results.values()) / 4
        color = "#2ecc71" if total > 7 else "#f1c40f" if total > 4 else "#e74c3c"
        self.final_score.configure(text=f"Total Score: {round(total, 1)} / 10", text_color=color)

if __name__ == "__main__":
    app = EssayScorerApp()
    app.mainloop()