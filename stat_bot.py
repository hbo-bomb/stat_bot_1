import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import subprocess
import os

MODEL_NAME = "de_core_news_sm"

# Function to download and verify the model
def ensure_spacy_model():
    model_dir = f"/home/adminuser/.local/lib/python3.12/site-packages/spacy/data/{MODEL_NAME}"

    # Force re-download if the model is missing
    if not os.path.exists(model_dir):
        st.warning(f"Model '{MODEL_NAME}' not found. Downloading now...")
        subprocess.run(["python", "-m", "spacy", "download", MODEL_NAME], check=True)

        # Set environment variable so SpaCy can find it
        os.environ["SPACY_DATA"] = "/home/adminuser/.local/lib/python3.12/site-packages/spacy/data"

    try:
        nlp = spacy.load(MODEL_NAME)
        st.success(f"SpaCy model '{MODEL_NAME}' loaded successfully!")
        return nlp
    except OSError as e:
        st.error(f"Failed to load the model. Error: {e}")
        st.stop()

# Load model
nlp = ensure_spacy_model()

st.success("SpaCy model loaded successfully!")



# Define keywords for each statistical test
keywords = {
    "z-test": ["Gauss-Test", "z-Test"],
    "t-test-1": ["1-Stichproben t-Test"],
    "t-test-2": ["2-Stichproben t-Test"],
    "wilcoxon": ["Wilcoxon-Test"],
    "anova-1": ["ANOVA einfaktoriell"],
    "chi-square": ["X^2-Test", "Chi-Quadrat-Test"],
    "spearman-correlation": ["Spearman-Korrelation"],
}

# Identify which statistical test is needed
def identify_test(text):
    doc = nlp(text)
    detected_tests = []
    for token in doc:
        for test, terms in keywords.items():
            if any(term.lower() in token.text.lower() for term in terms):
                detected_tests.append(test)
    return set(detected_tests)

# Statistical test functions
def perform_z_test(sample, pop_mean, pop_std):
    z_stat = (np.mean(sample) - pop_mean) / (pop_std / np.sqrt(len(sample)))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    explanation = (
        f"Der z-Test berechnet die Differenz zwischen dem Stichprobenmittelwert und dem Populationsmittelwert, "
        f"geteilt durch den Standardfehler ({pop_std}/sqrt(N)). Das ergibt den z-Wert {z_stat:.4f}. "
        f"Der p-Wert zeigt an, ob die Nullhypothese verworfen wird."
    )
    return z_stat, p_value, explanation

def perform_t_test_1sample(sample, pop_mean):
    t_stat, p_value = stats.ttest_1samp(sample, pop_mean)
    explanation = (
        f"Der t-Test vergleicht den Mittelwert der Stichprobe mit dem Populationsmittelwert. "
        f"Hier wurde ein t-Wert von {t_stat:.4f} berechnet, um die Nullhypothese zu testen."
    )
    return t_stat, p_value, explanation

def perform_t_test_2sample(sample1, sample2):
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
    explanation = (
        f"Ein 2-Stichproben t-Test vergleicht die Mittelwerte zweier unabhängiger Gruppen. "
        f"Der berechnete t-Wert ist {t_stat:.4f}, und der p-Wert zeigt, ob sich die Gruppen signifikant unterscheiden."
    )
    return t_stat, p_value, explanation

def perform_wilcoxon(sample1, sample2):
    stat, p_value = stats.wilcoxon(sample1, sample2)
    explanation = (
        f"Der Wilcoxon-Test ist ein nicht-parametrischer Test für gepaarte Stichproben. "
        f"Hier ergibt sich ein Wilcoxon-Statistik-Wert von {stat:.4f} mit einem p-Wert von {p_value:.4f}."
    )
    return stat, p_value, explanation

# Streamlit UI
st.title("📊 Erweiterter Statistik-Bot mit Erklärung")
st.write("Dieser Bot erkennt das Statistikproblem, erklärt den Lösungsweg und gibt die Berechnungsergebnisse aus.")

# User input for problem description
input_text = st.text_area("Geben Sie eine Beschreibung des Problems ein:")
detected_tests = identify_test(input_text)

if detected_tests:
    st.write(f"**Erkannte Tests:** {', '.join(detected_tests)}")
else:
    st.write("Kein spezifischer Test erkannt. Bitte geben Sie mehr Details ein.")

# Upload data file
uploaded_file = st.file_uploader("Laden Sie eine CSV-Datei mit Ihren Daten hoch", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file, sep=";")
    st.write("**Datenvorschau:**")
    st.write(data.head())

    # Perform tests based on detected problem
    try:
        if "z-test" in detected_tests:
            pop_mean = st.number_input("Geben Sie den Populationsmittelwert ein", value=0.0)
            pop_std = st.number_input("Geben Sie die Populationsstandardabweichung ein", value=1.0)
            if "sample" in data.columns:
                z_stat, p_value, explanation = perform_z_test(data["sample"], pop_mean, pop_std)
                st.write(f"### **Ergebnisse des Z-Tests:**")
                st.write(f"Z-Wert: {z_stat:.4f}")
                st.write(f"P-Wert: {p_value:.4f}")
                st.write("### **Erklärung des Lösungsweges:**")
                st.write(explanation)

        if "t-test-1" in detected_tests:
            pop_mean = st.number_input("Geben Sie den Populationsmittelwert ein", value=0.0)
            if "sample" in data.columns:
                t_stat, p_value, explanation = perform_t_test_1sample(data["sample"], pop_mean)
                st.write(f"### **Ergebnisse des t-Tests:**")
                st.write(f"T-Wert: {t_stat:.4f}")
                st.write(f"P-Wert: {p_value:.4f}")
                st.write("### **Erklärung des Lösungsweges:**")
                st.write(explanation)

    except Exception as e:
        st.error(f"Fehler bei der Analyse: {e}")
