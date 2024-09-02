import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

def is_grayscale(img_rgb):
    return np.allclose(img_rgb[:,:,0], img_rgb[:,:,1]) and np.allclose(img_rgb[:,:,1], img_rgb[:,:,2])

def measure_blur(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def analyze_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    is_gray = is_grayscale(img_rgb)

    color_means = np.mean(img_rgb, axis=(0,1))

    # Neue Berechnung für Sättigung und Helligkeit
    mean_sat, std_sat = cv2.meanStdDev(img_hsv[:,:,1])
    mean_bright, std_bright = cv2.meanStdDev(img_hsv[:,:,2])
    
    saturation_mean = mean_sat[0][0]
    saturation_std = std_sat[0][0]
    brightness_mean = mean_bright[0][0]
    brightness_std = std_bright[0][0]

    significant_saturation_threshold = 51  # 20% von 255
    significant_saturation_percentage = np.mean(img_hsv[:,:,1] > significant_saturation_threshold) * 100

    hue_mean = np.mean(img_hsv[:,:,0]) if not is_gray else 0

    shadows = np.mean(img_gray < 64)
    highlights = np.mean(img_gray > 192)

    blur_measure = measure_blur(img_gray)

    return {
        'filename': uploaded_file.name,
        'is_grayscale': is_gray,
        'r_mean': color_means[0],
        'g_mean': color_means[1],
        'b_mean': color_means[2],
        'brightness_mean': brightness_mean,
        'brightness_std': brightness_std,
        'saturation_mean': saturation_mean,
        'saturation_std': saturation_std,
        'significant_saturation_percentage': significant_saturation_percentage,
        'hue_mean': hue_mean,
        'shadow_proportion': shadows,
        'highlight_proportion': highlights,
        'blur_measure': blur_measure
    }

def calculate_consistency_score(df):
    features = {
        'color': ['r_mean', 'g_mean', 'b_mean'],
        'brightness': ['brightness_mean', 'brightness_std'],
        'saturation': ['saturation_mean', 'saturation_std', 'significant_saturation_percentage'],
        'shadows_highlights': ['shadow_proportion', 'highlight_proportion'],
        'contrast': ['brightness_std'],
        'blur': ['blur_measure']
    }
    
    weights = {
        'color': 0.05,
        'brightness': 0.20,
        'saturation': 0.10,
        'shadows_highlights': 0.25,
        'contrast': 0.25,
        'blur': 0.15
    }
    
    scores = {}
    
    for feature_group, feature_list in features.items():
        group_data = df[feature_list]
        
        if feature_group in ['brightness', 'saturation']:
            std_dev = np.std(group_data['brightness_mean' if feature_group == 'brightness' else 'saturation_mean'])
            score = 100 - std_dev
            scores[feature_group] = max(0, score)
        elif feature_group == 'contrast':
            cv = np.std(group_data['brightness_std']) / np.mean(group_data['brightness_std'])
            scores[feature_group] = 100 * np.exp(-3 * cv)
        else:
            cv = np.mean([np.std(group_data[col]) / (np.mean(group_data[col]) + 1e-5) for col in feature_list])
            raw_score = 100 * (1 - cv**2)
            scores[feature_group] = max(0, raw_score)
    
    weighted_score = sum(scores[group] * weight for group, weight in weights.items())
    
    return max(0, weighted_score), scores

def create_report(consistency_score, feature_scores, df):
    report = f"Visual Consistency Analysis Report\n\n"
    report += f"Overall Consistency Score: {consistency_score:.2f}%\n\n"
    report += "Feature Scores:\n"
    for feature, score in feature_scores.items():
        report += f"- {feature.capitalize()}: {score:.2f}%\n"
    report += "\nDetailed Analysis:\n"
    report += df.to_string()
    return report

def interpret_score(score):
    if score >= 90:
        return "Excellent! Your images have a high level of visual consistency."
    elif score >= 80:
        return "Very good. Your images show strong visual consistency with some room for improvement."
    elif score >= 70:
        return "Good. Your images have a decent level of consistency, but there's significant room for improvement."
    elif score >= 60:
        return "Fair. Your images show some consistency, but there's a lot of room for improvement."
    else:
        return "Needs improvement. Your images lack visual consistency and could benefit from a more cohesive approach."

def main():
     # Logo laden und anzeigen
    logo = Image.open('SE_Logo_Button_RGB-ON Blau.png')  # Ersetzen Sie dies mit dem tatsächlichen Pfad zu Ihrem Logo
    st.image(logo, width=200)
    
    st.set_page_config(page_title="Visual Consistency Challenge", page_icon="🎨")
    
    st.title("Visual Consistency Challenge")

    st.write("Bist du bereit, dein Auge für visuelle Konsistenz auf die Probe zu stellen? Lade 5 Bilder hoch und schätze ihre Konsistenz ein. Unser Tool wird dir zeigen, wie gut du wirklich bist!")

    st.write("Visuelle Konsistenz ist die geheime Geheimwaffe erfolgreicher Marken. Wer sie beherrscht, kann Kunden magisch anziehen und Konkurrenten in den Schatten stellen. Bist du bereit, deine Superkraft zu entdecken?")

    uploaded_files = st.file_uploader("Fordere dich heraus: Lade 5 Bilder hoch!", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        filenames = [file.name for file in uploaded_files]
        if len(filenames) != len(set(filenames)):
            st.warning("Achtung: Es wurden doppelte Dateien hochgeladen. Dies könnte das Ergebnis verfälschen.")
        
        if st.button("Zeig mir meine Superkraft!"):
            results = []
            for uploaded_file in uploaded_files:
                results.append(analyze_image(uploaded_file))
            
            df = pd.DataFrame(results)
            
            if any(df['is_grayscale']):
                st.warning("Achtung: Mindestens ein Schwarz-Weiß-Bild wurde erkannt. Dies kann die Farbkonsistenzanalyse beeinflussen.")
            
            consistency_score, feature_scores = calculate_consistency_score(df)
            
            if consistency_score < 20:
                st.warning("Achtung: Die analysierten Bilder zeigen eine sehr hohe Inkonsistenz!")
            
            st.markdown("<h2 style='text-align: center; color: #1E90FF;'>Dein Konsistenz-Radar Ergebnis</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; color: black;'>Gesamtkonsistenz: {consistency_score:.2f}%</h1>", unsafe_allow_html=True)
            
            interpretation = interpret_score(consistency_score)
            st.markdown(f"<h3 style='text-align: center; color: black;'>Interpretation: {interpretation}</h3>", unsafe_allow_html=True)
            
            feature_explanations = {
                'color': "Misst, wie einheitlich die Farbverteilung in den Bildern ist.",
                'brightness': "Zeigt, wie konsistent die Gesamthelligkeit der Bilder ist.",
                'saturation': "Bewertet die Konsistenz der Farbintensität und den Anteil signifikant gesättigter Bereiche in den Bildern.",
                'shadows_highlights': "Untersucht die Konsistenz von sehr dunklen und sehr hellen Bereichen.",
                'contrast': "Bewertet die Gleichmäßigkeit des Unterschieds zwischen hellen und dunklen Bereichen.",
                'blur': "Bewertet die Konsistenz des kreativen Einsatzes von Unschärfe in den Bildern."
            }

            for feature, score in feature_scores.items():
                st.markdown(f"<h3 style='color: black;'>{feature.capitalize()}: {score:.2f}%</h3>", unsafe_allow_html=True)
                st.write(feature_explanations[feature])
            
            st.write("Deine Bildauswahl:")
            cols = st.columns(4)
            for idx, uploaded_file in enumerate(uploaded_files):
                cols[idx % 4].image(uploaded_file, use_column_width=True)
            
            report_content = create_report(consistency_score, feature_scores, df)
            b64 = base64.b64encode(report_content.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="konsistenz_analyse.txt">Lade deine detaillierte Analyse herunter</a>'
            st.markdown(href, unsafe_allow_html=True)

    st.write("---")
    st.write("### Warum visuelle Konsistenz so wichtig ist:")
    st.write("- Sie schafft Wiedererkennungswert und stärkt deine Markenidentität.")
    st.write("- Sie erhöht das Vertrauen deiner Kunden in deine Professionalität.")
    st.write("- Sie macht deine Marketingbotschaften einprägsamer und effektiver.")
    st.write("- Sie hebt dich von der Konkurrenz ab und macht dich unverwechselbar.")
    st.write("Beherrsche die Kunst der visuellen Konsistenz, und deine Marke wird unaufhaltsam!")

if __name__ == "__main__":
    main()
