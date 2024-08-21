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

    brightness_channel = img_hsv[:,:,2]
    brightness_mean = np.mean(brightness_channel)
    brightness_std = np.std(brightness_channel)

    saturation_mean = np.mean(img_hsv[:,:,1]) if not is_gray else 0
    hue_mean = np.mean(img_hsv[:,:,0]) if not is_gray else 0

    threshold = np.percentile(img_gray, 90)
    mask = img_gray > threshold
    
    r_mean = np.mean(img_rgb[:,:,0][mask])
    g_mean = np.mean(img_rgb[:,:,1][mask])
    b_mean = np.mean(img_rgb[:,:,2][mask])
    
    wb_r = r_mean / (r_mean + g_mean + b_mean)
    wb_g = g_mean / (r_mean + g_mean + b_mean)
    wb_b = b_mean / (r_mean + g_mean + b_mean)
    
    wb_rg_ratio = np.exp(5 * (wb_r - wb_g))
    wb_rb_ratio = np.exp(5 * (wb_r - wb_b))

    wb_temp = 100 * (wb_rb_ratio - 1) / (wb_rb_ratio + 1)

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
        'hue_mean': hue_mean,
        'saturation_mean': saturation_mean,
        'wb_rg_ratio': wb_rg_ratio,
        'wb_rb_ratio': wb_rb_ratio,
        'wb_temp': wb_temp,
        'shadow_proportion': shadows,
        'highlight_proportion': highlights,
        'blur_measure': blur_measure
    }

def calculate_consistency_score(df):
    features = {
        'color': ['r_mean', 'g_mean', 'b_mean'],
        'brightness': ['brightness_mean', 'brightness_std'],
        'white_balance': ['wb_temp'],
        'shadows_highlights': ['shadow_proportion', 'highlight_proportion'],
        'contrast': ['brightness_std'],
        'blur': ['blur_measure']
    }
    
    weights = {
        'color': 0.05,
        'brightness': 0.20,
        'white_balance': 0.15,
        'shadows_highlights': 0.25,
        'contrast': 0.20,
        'blur': 0.15
    }
    
    scores = {}
    
    for feature_group, feature_list in features.items():
        group_data = df[feature_list]
        cv = np.mean([np.std(group_data[col]) / (np.mean(group_data[col]) + 1e-5) for col in feature_list])
        
        if feature_group == 'contrast':
            scores[feature_group] = 100 * np.exp(-3 * cv)
        else:
            raw_score = 100 * (1 - cv**2)
            scores[feature_group] = max(0, raw_score)
    
    weighted_score = sum(scores[group] * weight for group, weight in weights.items())
    
    return max(0, weighted_score), scores

def create_report(consistency_score, feature_scores, df):
    report_content = f"Dein Konsistenz-Radar Ergebnis: {consistency_score:.2f}%\n\n"
    
    feature_explanations = {
        'color': "Misst, wie einheitlich die Farbverteilung in den Bildern ist.",
        'brightness': "Zeigt, wie konsistent die Gesamthelligkeit der Bilder ist.",
        'white_balance': "Prüft, ob die Farbtemperatur in allen Bildern ähnlich ist.",
        'shadows_highlights': "Untersucht die Konsistenz von sehr dunklen und sehr hellen Bereichen.",
        'contrast': "Bewertet die Gleichmäßigkeit des Unterschieds zwischen hellen und dunklen Bereichen.",
        'blur': "Bewertet die Konsistenz des kreativen Einsatzes von Unschärfe in den Bildern."
    }

    for feature, score in feature_scores.items():
        report_content += f"{feature.capitalize()}: {score:.2f}%\n"
        report_content += f"{feature_explanations[feature]}\n\n"

    report_content += "Detaillierte Bildanalyse:\n"
    report_content += df.to_string()

    return report_content

def interpret_score(score):
    if score >= 90:
        return "Highly Consistent"
    elif score >= 80:
        return "Consistent"
    elif score >= 60:
        return "Moderately Consistent"
    elif score >= 40:
        return "Inconsistent"
    else:
        return "Highly Inconsistent"

def interpret_wb(temp):
    if temp > 50:
        return "sehr warm"
    elif temp > 25:
        return "warm"
    elif temp < -50:
        return "sehr kühl"
    elif temp < -25:
        return "kühl"
    else:
        return "neutral"

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
                'white_balance': "Prüft, ob die Farbtemperatur in allen Bildern ähnlich ist.",
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
