import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import io
import base64

def analyze_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    color_means = np.mean(img_rgb, axis=(0,1))

    brightness_mean = np.mean(img_gray)
    brightness_std = np.std(img_gray)

    saturation_mean = np.mean(img_hsv[:,:,1])

    wb_r = np.mean(img_rgb[:,:,0]) / brightness_mean
    wb_g = np.mean(img_rgb[:,:,1]) / brightness_mean
    wb_b = np.mean(img_rgb[:,:,2]) / brightness_mean

    edges = cv2.Canny(img_gray, 100, 200)
    edge_density = np.sum(edges) / (img.shape[0] * img.shape[1])

    shadows = np.mean(img_gray < 64)
    highlights = np.mean(img_gray > 192)

    return {
        'filename': uploaded_file.name,
        'r_mean': color_means[0],
        'g_mean': color_means[1],
        'b_mean': color_means[2],
        'brightness_mean': brightness_mean,
        'brightness_std': brightness_std,
        'saturation_mean': saturation_mean,
        'wb_r': wb_r,
        'wb_g': wb_g,
        'wb_b': wb_b,
        'edge_density': edge_density,
        'shadow_proportion': shadows,
        'highlight_proportion': highlights
    }

def calculate_consistency_score(df):
    features = {
        'color': ['r_mean', 'g_mean', 'b_mean'],
        'brightness': ['brightness_mean'],
        'saturation': ['saturation_mean'],
        'white_balance': ['wb_r', 'wb_g', 'wb_b'],
        'edge': ['edge_density'],
        'shadows_highlights': ['shadow_proportion', 'highlight_proportion'],
        'contrast': ['brightness_std']
    }
    
    weights = {
        'color': 0.25,
        'brightness': 0.20,
        'saturation': 0.15,
        'white_balance': 0.15,
        'edge': 0.10,
        'shadows_highlights': 0.10,
        'contrast': 0.05
    }
    
    scores = {}
    
    for feature_group, feature_list in features.items():
        group_data = df[feature_list]
        cv = np.mean([np.std(group_data[col]) / np.mean(group_data[col]) for col in feature_list])
        scores[feature_group] = np.exp(-cv) * 100  # Umwandlung in Prozent
    
    weighted_score = sum(scores[group] * weight for group, weight in weights.items())
    
    return weighted_score, scores

def create_pdf(consistency_score, feature_scores, df):
    # Da wir reportlab nicht verwenden, geben wir stattdessen einen String zurück
    pdf_content = f"Overall Consistency Score: {consistency_score:.2f}%\n\n"
    
    feature_explanations = {
        'color': "Misst, wie einheitlich die Farbverteilung in den Bildern ist.",
        'brightness': "Zeigt, wie konsistent die Gesamthelligkeit der Bilder ist.",
        'saturation': "Bewertet die Gleichmäßigkeit der Farbintensität über alle Bilder.",
        'white_balance': "Prüft, ob die Farbtemperatur in allen Bildern ähnlich ist.",
        'edge': "Vergleicht die Menge und Stärke von Konturen und Details in den Bildern.",
        'shadows_highlights': "Untersucht die Konsistenz von sehr dunklen und sehr hellen Bereichen.",
        'contrast': "Bewertet die Gleichmäßigkeit des Unterschieds zwischen hellen und dunklen Bereichen."
    }

    for feature, score in feature_scores.items():
        pdf_content += f"{feature.capitalize()}: {score:.2f}%\n"
        pdf_content += f"{feature_explanations[feature]}\n\n"

    pdf_content += "Image Analysis Results:\n"
    pdf_content += df.to_string()

    return pdf_content

def main():

    # Logo laden und anzeigen
    logo = Image.open('SE_Logo_Button_RGB-ON Blau.png')  # Ersetzen Sie dies mit dem tatsächlichen Pfad zu Ihrem Logo
    st.image(logo, width=200)

    st.title("Image Consistency Analyzer")

    uploaded_files = st.file_uploader("Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files and st.button("Analyze Images"):
        results = []
        for uploaded_file in uploaded_files:
            results.append(analyze_image(uploaded_file))
        
        df = pd.DataFrame(results)
        
        consistency_score, feature_scores = calculate_consistency_score(df)
        
        st.markdown(f"<h1 style='text-align: center; color: black;'>Overall Consistency Score: {consistency_score:.2f}%</h1>", unsafe_allow_html=True)
        
        feature_explanations = {
            'color': "Misst, wie einheitlich die Farbverteilung in den Bildern ist.",
            'brightness': "Zeigt, wie konsistent die Gesamthelligkeit der Bilder ist.",
            'saturation': "Bewertet die Gleichmäßigkeit der Farbintensität über alle Bilder.",
            'white_balance': "Prüft, ob die Farbtemperatur in allen Bildern ähnlich ist.",
            'edge': "Vergleicht die Menge und Stärke von Konturen und Details in den Bildern.",
            'shadows_highlights': "Untersucht die Konsistenz von sehr dunklen und sehr hellen Bereichen.",
            'contrast': "Bewertet die Gleichmäßigkeit des Unterschieds zwischen hellen und dunklen Bereichen."
        }

        for feature, score in feature_scores.items():
            st.markdown(f"<h3 style='color: black;'>{feature.capitalize()}: {score:.2f}%</h3>", unsafe_allow_html=True)
            st.write(feature_explanations[feature])
        
        st.write("Uploaded Images:")
        cols = st.columns(4)
        for idx, uploaded_file in enumerate(uploaded_files):
            cols[idx % 4].image(uploaded_file, use_column_width=True)
        
        pdf_content = create_pdf(consistency_score, feature_scores, df)
        b64 = base64.b64encode(pdf_content.encode()).decode()
        href = f'<a href="data:text/plain;base64,{b64}" download="consistency_analysis.txt">Download Results as Text File</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
