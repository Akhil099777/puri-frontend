import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from pathlib import Path
import plotly.express as px
# OCR & PDF tools
import pytesseract
from PIL import Image
import pdfplumber
from pdf2image import convert_from_path

# ✅ Point pytesseract to your tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =============================
# OCR Helper Functions
# =============================
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF (works for both text-PDF and scanned)."""
    text = ""

    try:
        # Try text-based extraction
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except:
        pass

    # If no text found → use OCR on images
    if not text.strip():
        images = convert_from_path(pdf_file)
        for img in images:
            text += pytesseract.image_to_string(img)

    return text


def extract_text_from_image(image_file):
    """Extract text from uploaded image."""
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)
    return text

# PDF report packages
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# =============================
# Load Model & Encoder
# =============================
root = Path(__file__).parent
model_path = root / "models" / "symptom_disease_model.joblib"
encoder_path = root / "models" / "symptom_encoder.joblib"
diabetes_model_path = root / "models" / "diabetes_model.joblib"

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)
diabetes_model = joblib.load(diabetes_model_path)

# Symptom list
symptom_list = encoder.classes_
# Symptom list
symptom_list = encoder.classes_

# =============================
# Symptom Synonyms Mapping
# (Place this here, so it's defined before any tab uses it)
# =============================
# -----------------------------
# Extended Symptom Synonyms Dictionary
# -----------------------------
SYMPTOM_SYNONYMS = {
    "fever": ["fever", "high temperature", "pyrexia", "raised body temp", "temperature"],
    "cough": ["cough", "dry cough", "persistent cough", "whooping cough"],
    "headache": ["headache", "migraine", "head pain", "pressure in head"],
    "fatigue": ["fatigue", "tiredness", "weakness", "exhaustion", "lethargy"],
    "nausea": ["nausea", "vomiting", "queasiness", "feeling sick"],
    "chest pain": ["chest pain", "chest discomfort", "tightness in chest", "angina"],
    "shortness of breath": [
        "shortness of breath", "breathlessness", "difficulty breathing", "dyspnea"
    ],
    "sore throat": ["sore throat", "throat pain", "throat discomfort", "pharyngitis"],
    "diarrhea": ["diarrhea", "loose stools", "frequent bowel movement", "watery stools"],
    "constipation": ["constipation", "hard stools", "difficulty passing stool"],
    "dizziness": ["dizziness", "lightheadedness", "vertigo", "giddiness"],
    "abdominal pain": ["abdominal pain", "stomach pain", "belly ache", "gastric pain"],
    "loss of appetite": ["loss of appetite", "poor appetite", "no desire to eat"],
    "joint pain": ["joint pain", "arthritis", "aching joints", "joint stiffness"],
    "back pain": ["back pain", "lower back pain", "spinal pain"],
    "rash": ["rash", "skin eruption", "skin spots", "dermatitis"],
    "weight loss": ["weight loss", "unintentional weight loss", "losing weight"],
    "weight gain": ["weight gain", "obesity", "overweight"],
    "anemia": ["anemia", "low hemoglobin", "pale skin", "iron deficiency"],
    "swelling": ["swelling", "edema", "inflammation", "fluid retention"],
    "palpitations": ["palpitations", "irregular heartbeat", "rapid heartbeat"],
    "blurred vision": ["blurred vision", "loss of vision", "dim vision"],
    "chills": ["chills", "shivering", "rigors"],
    "night sweats": ["night sweats", "sweating at night"],
    "anxiety": ["anxiety", "nervousness", "restlessness"],
    "depression": ["depression", "sadness", "low mood"],
    "confusion": ["confusion", "disorientation", "mental fog"],
    "runny nose": ["runny nose", "nasal discharge", "rhinitis"],
    "sneezing": ["sneezing", "nasal irritation"],
    "itching": ["itching", "pruritus", "skin irritation"],
    "yellowing of eyes": ["yellowing of eyes", "jaundice", "yellow sclera"],
    "burning urination": ["burning urination", "painful urination", "dysuria"],
    "increased urination": ["increased urination", "frequent urination", "polyuria"],
    "thirst": ["thirst", "excessive thirst", "polydipsia"],
    "loss of consciousness": ["loss of consciousness", "fainting", "blackout"],
    "seizures": ["seizures", "fits", "convulsions"],
    "swollen lymph nodes": ["swollen lymph nodes", "enlarged glands", "lymphadenopathy"],
    "bleeding gums": ["bleeding gums", "gum bleeding", "gingival bleeding"],
    "mouth ulcers": ["mouth ulcers", "canker sores", "oral ulcers"],
    "hair loss": ["hair loss", "alopecia", "hair thinning"],
    "cold hands and feet": ["cold hands and feet", "poor circulation"],
    "excessive sweating": ["excessive sweating", "hyperhidrosis"],
    "hearing loss": ["hearing loss", "deafness", "reduced hearing"],
    "ear pain": ["ear pain", "earache", "otalgia"],
    "eye pain": ["eye pain", "ocular pain"],
    "insomnia": ["insomnia", "sleeplessness", "trouble sleeping"],
    "swollen ankles": ["swollen ankles", "ankle edema"],
    "bleeding": ["bleeding", "hemorrhage", "blood loss"],
    "weakness": ["weakness", "lack of strength", "frailty"],
    "cold": ["cold", "common cold", "flu-like symptoms"]
}



# =============================
# PDF Report Generator
# =============================
def generate_report(patient, symptoms, results, filename="report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Healthcare Disease Risk Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # Patient Details
    story.append(Paragraph("<b>Patient Details</b>", styles["Heading2"]))
    details = f"""
    <b>Name:</b> {patient.get("name", "N/A")} <br/>
    <b>Age:</b> {patient.get("age", "N/A")} <br/>
    <b>Gender:</b> {patient.get("gender", "N/A")} <br/>
    <b>Medical History:</b> {patient.get("medical_history", "N/A")}
    """
    story.append(Paragraph(details, styles["Normal"]))
    story.append(Spacer(1, 12))

    # Symptoms
    story.append(Paragraph("<b>Reported Symptoms</b>", styles["Heading2"]))
    story.append(Paragraph(", ".join(symptoms) if symptoms else "None", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Prediction Results (top 5)
    story.append(Paragraph("<b>Prediction Results</b>", styles["Heading2"]))
    table_data = [["Disease", "Probability"]]
    for _, row in results.head(5).iterrows():
        table_data.append([row["Disease"], f"{row['Probability']:.2f}"])

    table = Table(table_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4B8BBE")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Note
    story.append(Paragraph("<b>Doctor's Note</b>", styles["Heading2"]))
    story.append(Paragraph(
        "This report is AI-generated based on selected symptoms. "
        "It is not a medical diagnosis. Please consult a certified doctor for confirmation.",
        styles["Italic"]
    ))

    doc.build(story)
    return filename

# =============================
# Prediction Helper
# =============================
def predict_disease(symptoms):
    X_input = np.zeros(len(symptom_list))
    for s in symptoms:
        if s in symptom_list:
            idx = np.where(symptom_list == s)[0][0]
            X_input[idx] = 1
    preds = model.predict_proba([X_input])[0]
    results = pd.DataFrame({"Disease": model.classes_, "Probability": preds})
    results = results.sort_values(by="Probability", ascending=False).reset_index(drop=True)
    return results

def plot_probabilities(results, top_n=10):
    fig = px.bar(
        results.head(top_n),
        x="Probability",
        y="Disease",
        orientation="h",
        title="Top Predicted Diseases",
        text="Probability"
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Comparative Analysis of ML Models for Multi-Disease Prediction in Healthcare", layout="wide")

st.title("🏥Comparative Analysis of ML Models for Multi-Disease Prediction in Healthcare")

tabs = st.tabs([
    "Single Prediction",
    "Batch CSV Prediction",
    "Symptom-based Prediction",
    "Diabetes Prediction",
    "Upload Medical Report"
])

# -----------------------------
# Tab 1: Single Prediction
# -----------------------------
with tabs[0]:
    st.header("Patient Details")

    with st.form("single_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with col2:
            medical_history = st.text_area("Medical History", placeholder="e.g., Hypertension, past surgeries")

        st.subheader("Current Symptoms")
        symptoms = st.multiselect("Select symptoms", options=symptom_list)

        submitted = st.form_submit_button("🔍 Predict Disease")

    if submitted:
        if symptoms:
            results = predict_disease(symptoms)
            st.subheader("Prediction Results")
            st.dataframe(results.head(10))

            fig = plot_probabilities(results)
            st.plotly_chart(fig, use_container_width=True)

            patient_info = {
                "name": name,
                "age": age,
                "gender": gender,
                "medical_history": medical_history
            }
            report_path = root / "reports"
            os.makedirs(report_path, exist_ok=True)
            pdf_file = report_path / f"{name or 'patient'}_report.pdf"

            generate_report(patient_info, symptoms, results, filename=str(pdf_file))

            with open(pdf_file, "rb") as f:
                st.download_button("📄 Download Report", f, file_name=pdf_file.name, mime="application/pdf")

            # -----------------------------
            # ✅ Save prediction to patient_history.csv
            # -----------------------------
            history_file = "patient_history.csv"
            top_disease = results.iloc[0]["Disease"]
            top_prob = results.iloc[0]["Probability"]

            new_record = pd.DataFrame([{
                "Patient Name": name or "Unknown",
                "Age": age,
                "Gender": gender,
                "Medical History": medical_history,
                "Symptoms": ", ".join(symptoms),
                "Top Predicted Disease": top_disease,
                "Probability": round(top_prob, 2)
            }])

            if os.path.exists(history_file):
                old = pd.read_csv(history_file)
                updated = pd.concat([old, new_record], ignore_index=True)
                updated.to_csv(history_file, index=False)
            else:
                new_record.to_csv(history_file, index=False)

        else:
            st.warning("Please select at least one symptom.")

    # -----------------------------
    # 📜 Patient History Button
    # -----------------------------
    st.markdown("---")
    if st.button("📜 View Patient History"):
        history_file = "patient_history.csv"

        try:
            df_history = pd.read_csv(history_file)

            if not df_history.empty:
                st.subheader("Saved Patient Records")
                st.dataframe(df_history)

                # Search by name
                patient_filter = st.text_input("🔎 Search by Patient Name")
                if patient_filter:
                    filtered = df_history[df_history["Patient Name"].str.contains(patient_filter, case=False, na=False)]
                    st.dataframe(filtered)

            else:
                st.info("No patient history records available yet.")

        except FileNotFoundError:
            st.info("No patient history found. Make a prediction first.")


# -----------------------------
# Tab 2: Batch CSV Prediction
# -----------------------------
with tabs[1]:
    st.header("Batch CSV Prediction")
    st.write("Upload a CSV with columns **Name** and **Symptoms** (comma-separated).")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = [col.strip() for col in df.columns]

        if "Symptoms" not in df.columns:
            st.error("CSV must contain a 'Symptoms' column.")
        else:
            all_results = []
            for idx, row in enumerate(df.itertuples(index=False), start=1):
                patient_name = getattr(row, "Name") if "Name" in df.columns else f"Patient {idx}"
                symptoms_raw = str(getattr(row, "Symptoms"))
                symptoms = [s.strip() for s in symptoms_raw.split(",") if s.strip()]
                results = predict_disease(symptoms)
                top_disease = results.iloc[0]["Disease"]
                prob = results.iloc[0]["Probability"]

                all_results.append({
                    "Name": patient_name,
                    "Symptoms": ", ".join(symptoms),
                    "Top Disease": top_disease,
                    "Probability": round(prob, 2)
                })

            batch_df = pd.DataFrame(all_results)
            st.subheader("Batch Prediction Results")
            st.dataframe(batch_df)

# -----------------------------
# Tab 3: Symptom-based Prediction
# -----------------------------
with tabs[2]:
    st.header("Symptom-based Prediction")
    symptoms = st.multiselect("Select Symptoms", options=symptom_list)

    if st.button("🔍 Predict"):
        if symptoms:
            results = predict_disease(symptoms)
            st.subheader("Prediction Results")
            st.dataframe(results.head(10))

            fig = plot_probabilities(results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one symptom.")

# -----------------------------
# Tab 4: Diabetes Prediction
# -----------------------------
with tabs[3]:
    st.header("🩺 Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", min_value=0, step=1, key="pregnancies")
    glucose = st.number_input("Glucose Level", min_value=0, key="glucose")
    blood_pressure = st.number_input("Blood Pressure", min_value=0, key="blood_pressure")
    skin_thickness = st.number_input("Skin Thickness", min_value=0, key="skin_thickness")
    insulin = st.number_input("Insulin Level", min_value=0, key="insulin")
    bmi = st.number_input("BMI", min_value=0.0, key="bmi")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, key="dpf")
    age_diabetes = st.number_input("Age", min_value=1, key="age_diabetes")

    if st.button("🔍 Predict Diabetes", key="predict_diabetes"):
        features = np.array([[pregnancies, glucose, blood_pressure,
                              skin_thickness, insulin, bmi, dpf, age_diabetes]])
        prediction = diabetes_model.predict(features)[0]
        st.success("✅ Positive for Diabetes" if prediction == 1 else "❌ Negative for Diabetes")

# -----------------------------
# Tab 5: Upload Medical Report
# -----------------------------
with tabs[4]:
    st.header("📑 Upload Medical Report (PDF/Image)")

    uploaded_report = st.file_uploader("Upload Report", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_report is not None:
        file_type = uploaded_report.type

        if "pdf" in file_type:
            text = extract_text_from_pdf(uploaded_report)
        else:
            text = extract_text_from_image(uploaded_report)

        st.subheader("📄 Extracted Report Text")
        st.text_area("Report Content", text, height=300)

        if st.button("🔍 Analyze Report for Diseases"):
            found_symptoms = []
            lower_text = text.lower()

            # ✅ Match extracted text with synonyms
            for main_symptom, synonyms in SYMPTOM_SYNONYMS.items():
                for synonym in synonyms:
                    if synonym in lower_text:
                        found_symptoms.append(main_symptom)
                        break  # stop checking once one synonym matches

            if found_symptoms:
                st.write("✅ Symptoms found in report:", ", ".join(set(found_symptoms)))
                results = predict_disease(list(set(found_symptoms)))
                st.subheader("Prediction Results")
                st.dataframe(results.head(10))

                fig = plot_probabilities(results)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No recognizable symptoms found in the uploaded report.")


