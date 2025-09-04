from flask import Flask, render_template, request, send_file, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from fpdf import FPDF
import os
import datetime
import tempfile

app = Flask(__name__)

# ------------------------- Visitor Count -------------------------
visitor_count = 0

# ------------------------- Load Model -------------------------
def load_mammo_model():
    model_path = os.path.expanduser("./mammogram_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return load_model(model_path)

model = load_mammo_model()

# ------------------------- Prediction Logic -------------------------
def preprocess_and_predict(image: Image.Image):
    image = image.convert("L")
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 256, 256, 1)
    prediction = model.predict(img_array)[0][0]
    label = "Malignant (Cancerous)" if prediction > 0.5 else "Benign (Non-cancerous)"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, f"{confidence:.2%}"

# ------------------------- PDF Report -------------------------
def generate_pdf(label, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Mammogram Cancer Prediction Report", ln=1, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Prediction Result: {label}", ln=1)
    pdf.cell(200, 10, txt=f"Confidence Score: {confidence}", ln=2)
    pdf.cell(200, 10, txt=f"Date Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=3)

    temp_dir = tempfile.gettempdir()
    report_path = os.path.join(temp_dir, "prediction_report.pdf")
    pdf.output(report_path)
    return report_path

# ------------------------- Routes -------------------------
@app.route("/")
def home():
    global visitor_count
    visitor_count += 1
    return render_template("index.html", visitors=visitor_count)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("home"))
    
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("home"))
    
    image = Image.open(file.stream)
    label, confidence = preprocess_and_predict(image)
    
    pdf_path = generate_pdf(label, confidence)
    
    return render_template("result.html", label=label, confidence=confidence, pdf_url="/download_report")

@app.route("/download_report")
def download_report():
    temp_dir = tempfile.gettempdir()
    report_path = os.path.join(temp_dir, "prediction_report.pdf")
    return send_file(report_path, as_attachment=True, download_name="prediction_report.pdf")

@app.route("/recommendations")
def recommendations():
    return render_template("recommendations.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html", visitors=visitor_count)

# ------------------------- Run -------------------------
if __name__ == "__main__":
    app.run(debug=True)
