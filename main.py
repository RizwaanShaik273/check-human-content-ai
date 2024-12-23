import streamlit as st
import pandas as pd
from transformers import pipeline

# Set Streamlit page configuration
st.set_page_config(page_title="AI Detection Job Applicant Scoring", layout="wide")

# Allow user to select model and input custom labels
model_option = st.selectbox("Choose the classification model", ["roberta-large-mnli", "bert-base-uncased"])
custom_labels_input = st.text_area("Enter Custom Labels (comma-separated)", "AI, Human")
custom_labels = [label.strip() for label in custom_labels_input.split(",")]

confidence_threshold = st.slider("Set Confidence Threshold", 0.0, 1.0, 0.6, 0.01)

# Load the selected model
@st.cache_resource  # Updated caching decorator
def load_model_based_on_option(model_name):
    try:
        return pipeline("zero-shot-classification", model=model_name)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

classifier = load_model_based_on_option(model_option)

# Function to classify text with custom labels and thresholds
def classify_text_with_threshold(text, classifier, candidate_labels, threshold):
    if not text or len(text.strip()) == 0:
        st.warning("Empty or missing answer detected.")
        return "Empty"
    
    try:
        result = classifier(text, candidate_labels=candidate_labels)
        label = result["labels"][0]
        confidence = result["scores"][0]

        if confidence < threshold:
            return "Uncertain"
        return label
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return "Error"

# Function to process the CSV file and apply scoring logic
def process_csv(file, classifier, candidate_labels, threshold):
    try:
        # Attempt to read the file with utf-8 encoding, fallback to 'ISO-8859-1'
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding="ISO-8859-1")
        
        required_columns = ['email', 'a', 'b', 'c', 'd', 'e']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"CSV is missing columns: {', '.join(missing_columns)}")
            return None
        
        results = []
        for index, row in df.iterrows():
            total_score = 0
            email = row['email']
            
            # Loop through answers in columns 'a' to 'e'
            for col in required_columns[1:]:
                answer = row.get(col, "")
                answer_type = classify_text_with_threshold(answer, classifier, candidate_labels, threshold)
                
                if answer_type == "Human":
                    total_score += 2  # Human gets higher score
                elif answer_type == "AI":
                    total_score += 1  # AI gets lower score
                else:
                    total_score += 0  # Uncertain answers get no score

            results.append({"email": email, "total_score": total_score})

        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None

# Allow the user to upload the CSV and process it
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    with st.spinner("Processing your data... Please wait."):
        result_df = process_csv(uploaded_file, classifier, custom_labels, confidence_threshold)
        if result_df is not None:
            st.success("Analysis complete!")
            st.dataframe(result_df)

            # Option to download results
            csv_download = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", data=csv_download, file_name="applicant_scores.csv", mime="text/csv")
