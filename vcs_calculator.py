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
        'shadow_proportion': shadows,
        'highlight_proportion': highlights
    }

def calculate_consistency_score(df):
    features = {
        'color': ['r_mean', 'g_mean', 'b_mean'],
        'brightness': ['brightness_mean'],
        'saturation': ['saturation_mean'],
        'white_balance': ['wb_r', 'wb_g', 'wb_b'],
        'shadows_highlights': ['shadow_proportion', 'highlight_proportion'],
        'contrast': ['brightness_std']
    }
    
    weights = {
        'color': 0.25,
        'brightness': 0.20,
        'saturation': 0.15,
        'white_balance': 0.05,
        'shadows_highlights': 0.25,
        'contrast': 0.10
    }
    
    scores = {}
    
    for feature_group, feature_list in features.items():
        group_data = df[feature_list]
        cv = np.mean([np.std(group_data[col]) / (np.mean(group_data[col]) + 1e-5) for col in feature_list])
        
        if feature_group == 'saturation':
            scores[feature_group] = np.exp(-cv) * 100
        elif feature_group == 'contrast':
            scores[feature_group] = 100 * np.exp(-3 * cv)
        else:
            scores[feature_group] = 100 * (1 - cv**2)
    
    weighted_score = sum(scores[group] * weight for group, weight in weights.items())
    
    return weighted_score, scores

def create_report(consistency_score, feature_scores, df):
    report_content = f"Deine Konsistenz Kompetenz liegt bei: {consistency_score:.2f}%\n\n"
    
    feature_explanations = {
        'color': "Misst, wie einheitlich die Farbverteilung in den Bildern ist.",
        'brightness': "Zeigt, wie konsistent die Gesamthelligkeit der Bilder ist.",
        'saturation': "Bewertet die Gleichmäßigkeit der Farbintensität über alle Bilder.",
        'white_balance': "Prüft, ob die Farbtemperatur in allen Bildern ähnlich ist.",
        'shadows_highlights': "Untersucht die Konsistenz von sehr dunklen und sehr hellen Bereichen.",
        'contrast': "Bewertet die Gleichmäßigkeit des Unterschieds zwischen hellen und dunklen Bereichen."
    }

    for feature, score in feature_scores.items():
        report_content += f"{feature.capitalize()}: {score:.2f}%\n"
        report_content += f"{feature_explanations[feature]}\n\n"

    report_content += "Detaillierte Bildanalyse:\n"
    report_content += df.to_string()

    return report_content

def main():
    
    # Logo laden und anzeigen
    logo = Image.open('SE_Logo_Button_RGB-ON Blau.png')  # Ersetzen Sie dies mit dem tatsächlichen Pfad zu Ihrem Logo
    st.image(logo, width=200)

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
            
            consistency_score, feature_scores = calculate_consistency_score(df)
            
            st.markdown("<h2 style='text-align: center; color: #1E90FF;'>Dein Konsistenz-Radar Ergebnis</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; color: black;'>Gesamtkonsistenz: {consistency_score:.2f}%</h1>", unsafe_allow_html=True)
            
            feature_explanations = {
                'color': "Misst, wie einheitlich die Farbverteilung in den Bildern ist.",
                'brightness': "Zeigt, wie konsistent die Gesamthelligkeit der Bilder ist.",
                'saturation': "Bewertet die Gleichmäßigkeit der Farbintensität über alle Bilder.",
                'white_balance': "Prüft, ob die Farbtemperatur in allen Bildern ähnlich ist.",
                'shadows_highlights': "Untersucht die Konsistenz von sehr dunklen und sehr hellen Bereichen.",
                'contrast': "Bewertet die Gleichmäßigkeit des Unterschieds zwischen hellen und dunklen Bereichen."
            }

            for feature, score in feature_scores.items():
                st.markdown(f"<h3 style='color: black;'>{feature.capitalize()}: {score:.2f}%</h3>", unsafe_allow_html=True)
                st.write(feature_explanations[feature])
            
            st.write("Untersuchte Bilder:")
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
