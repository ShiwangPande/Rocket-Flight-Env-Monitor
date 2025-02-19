import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import io
import base64
from fpdf import FPDF
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for development

# IMPORTANT CSV columns from Arduino
IMPORTANT_COLS = [
    "Timestamp (HH:MM:SS)",
    "DHT_Temperature (C)",
    "DHT_Humidity (%)",
    "BMP_Temperature (C)",
    "BMP_Pressure (hPa)",
    "BMP_Altitude (m)",
    "Accel_X (m/s^2)",
    "Accel_Y (m/s^2)",
    "Accel_Z (m/s^2)",
    "Dust (ug/m^3)",
    "PM2.5 (ug/m^3)"
]

def convert_np(obj):
    """
    Recursively convert numpy data types in obj to native Python types.
    Convert any NaN values to None.
    """
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(x) for x in obj]
    else:
        return obj

def filter_dict(data: dict, keys: list):
    """Return a new dictionary including only the specified keys."""
    return {k: data[k] for k in keys if k in data}

def filter_preview(preview: list, keys: list):
    """Filter each row in preview to include only the specified keys."""
    return [{k: row[k] for k in keys if k in row} for row in preview]

# Calculate AQI from PM2.5 concentration (µg/m³) using US EPA breakpoints.
def calculateAQI(pm25):
    if pm25 < 0:
        return None
    if pm25 <= 12.0:
        aqi = (50.0/12.0) * pm25
    elif pm25 <= 35.4:
        aqi = ((100.0-51.0)/(35.4-12.1)) * (pm25-12.1) + 51.0
    elif pm25 <= 55.4:
        aqi = ((150.0-101.0)/(55.4-35.5)) * (pm25-35.5) + 101.0
    elif pm25 <= 150.4:
        aqi = ((200.0-151.0)/(150.4-55.5)) * (pm25-55.5) + 151.0
    elif pm25 <= 250.4:
        aqi = ((300.0-201.0)/(250.4-150.5)) * (pm25-150.5) + 201.0
    elif pm25 <= 350.4:
        aqi = ((400.0-301.0)/(350.4-250.5)) * (pm25-250.5) + 301.0
    elif pm25 <= 500.4:
        aqi = ((500.0-401.0)/(500.4-350.5)) * (pm25-350.5) + 401.0
    else:
        aqi = None
    return aqi

def generate_chart(df, x_col, y_cols, title, xlabel, ylabel, markers, legend_labels):
    """
    Generate a chart for given x and y columns.
    Returns a base64-encoded PNG image string.
    """
    plt.figure(figsize=(8, 4))
    for y, marker, label in zip(y_cols, markers, legend_labels):
        plt.plot(df[x_col], df[y], marker=marker, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_all_charts(df):
    """
    Generate multiple charts from the dataframe and return them in a dictionary.
    """
    charts = {}
    if "Timestamp (HH:MM:SS)" not in df.columns:
        return charts

    df_chart = df.head(50)

    # PM2.5 & AQI Chart
    if "PM2.5 (ug/m^3)" in df.columns:
        df_chart["PM2.5_AQI"] = df_chart["PM2.5 (ug/m^3)"].apply(calculateAQI)
        charts["pm25_chart"] = generate_chart(
            df_chart,
            "Timestamp (HH:MM:SS)",
            ["PM2.5 (ug/m^3)", "PM2.5_AQI"],
            "PM2.5 Concentration & AQI Over Time",
            "Time (HH:MM:SS)",
            "Value",
            markers=["o", "x"],
            legend_labels=["PM2.5 (ug/m^3)", "PM2.5 AQI"]
        )

    # Temperature Chart (DHT & BMP)
    if "DHT_Temperature (C)" in df.columns and "BMP_Temperature (C)" in df.columns:
        charts["temperature_chart"] = generate_chart(
            df_chart,
            "Timestamp (HH:MM:SS)",
            ["DHT_Temperature (C)", "BMP_Temperature (C)"],
            "Temperature Over Time",
            "Time (HH:MM:SS)",
            "Temperature (°C)",
            markers=["o", "x"],
            legend_labels=["DHT Temp", "BMP Temp"]
        )

    # Humidity Chart (DHT)
    if "DHT_Humidity (%)" in df.columns:
        charts["humidity_chart"] = generate_chart(
            df_chart,
            "Timestamp (HH:MM:SS)",
            ["DHT_Humidity (%)"],
            "Humidity Over Time",
            "Time (HH:MM:SS)",
            "Humidity (%)",
            markers=["o"],
            legend_labels=["DHT Humidity"]
        )

    # Pressure Chart (BMP)
    if "BMP_Pressure (hPa)" in df.columns:
        charts["pressure_chart"] = generate_chart(
            df_chart,
            "Timestamp (HH:MM:SS)",
            ["BMP_Pressure (hPa)"],
            "Pressure Over Time",
            "Time (HH:MM:SS)",
            "Pressure (hPa)",
            markers=["o"],
            legend_labels=["BMP Pressure"]
        )

    # Altitude Chart (BMP)
    if "BMP_Altitude (m)" in df.columns:
        charts["altitude_chart"] = generate_chart(
            df_chart,
            "Timestamp (HH:MM:SS)",
            ["BMP_Altitude (m)"],
            "Altitude Over Time",
            "Time (HH:MM:SS)",
            "Altitude (m)",
            markers=["o"],
            legend_labels=["BMP Altitude"]
        )

    # Accelerometer Chart (X, Y, Z) and Magnitude
    if all(col in df.columns for col in ["Accel_X (m/s^2)", "Accel_Y (m/s^2)", "Accel_Z (m/s^2)"]):
        charts["accel_chart"] = generate_chart(
            df_chart,
            "Timestamp (HH:MM:SS)",
            ["Accel_X (m/s^2)", "Accel_Y (m/s^2)", "Accel_Z (m/s^2)"],
            "Accelerometer Readings Over Time",
            "Time (HH:MM:SS)",
            "Acceleration (m/s^2)",
            markers=["o", "x", "s"],
            legend_labels=["Accel X", "Accel Y", "Accel Z"]
        )
        df_chart["Accel_Magnitude"] = np.sqrt(
            df_chart["Accel_X (m/s^2)"]**2 +
            df_chart["Accel_Y (m/s^2)"]**2 +
            df_chart["Accel_Z (m/s^2)"]**2
        )
        charts["accel_magnitude_chart"] = generate_chart(
            df_chart,
            "Timestamp (HH:MM:SS)",
            ["Accel_Magnitude"],
            "Acceleration Magnitude Over Time",
            "Time (HH:MM:SS)",
            "Magnitude (m/s^2)",
            markers=["o"],
            legend_labels=["Accel Magnitude"]
        )

    # Dust Chart
    if "Dust (ug/m^3)" in df.columns:
        charts["dust_chart"] = generate_chart(
            df_chart,
            "Timestamp (HH:MM:SS)",
            ["Dust (ug/m^3)"],
            "Dust Concentration Over Time",
            "Time (HH:MM:SS)",
            "Dust (ug/m^3)",
            markers=["o"],
            legend_labels=["Dust"]
        )

    # Pressure vs Altitude (Scatter)
    if "BMP_Pressure (hPa)" in df.columns and "BMP_Altitude (m)" in df.columns:
        plt.figure(figsize=(8, 4))
        plt.scatter(df_chart["BMP_Altitude (m)"], df_chart["BMP_Pressure (hPa)"], c='blue', marker='o')
        plt.xlabel("Altitude (m)")
        plt.ylabel("Pressure (hPa)")
        plt.title("Pressure vs Altitude")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        charts["pressure_altitude_chart"] = base64.b64encode(buf.getvalue()).decode("utf-8")

    return charts

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        df = pd.read_csv(file)
        if all(col in df.columns for col in IMPORTANT_COLS):
            df_filtered = df[IMPORTANT_COLS]
        else:
            df_filtered = df

        basic_summary = df_filtered.describe(include='all').to_dict()
        preview = df_filtered.head(10).to_dict(orient='records')
        charts = generate_all_charts(df_filtered)

        analysis = {
            'message': 'File processed successfully',
            'basic_summary': convert_np(filter_dict(basic_summary, IMPORTANT_COLS)),
            'data_preview': convert_np(filter_preview(preview, IMPORTANT_COLS)),
            'charts': charts
        }
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download', methods=['POST'])
def download_pdf():
    # Generate a PDF report containing summary, preview and charts.
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        df = pd.read_csv(file)
        if all(col in df.columns for col in IMPORTANT_COLS):
            df_filtered = df[IMPORTANT_COLS]
        else:
            df_filtered = df

        basic_summary = df_filtered.describe(include='all').to_dict()
        preview = df_filtered.head(10).to_dict(orient='records')
        charts = generate_all_charts(df_filtered)

        # Create PDF report using FPDF.
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Rocket Flight Environmental Report", ln=1, align="C")
        pdf.ln(10)
        
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, "Summary Statistics:", ln=1)
        for col, stats in basic_summary.items():
            pdf.cell(0, 10, f"{col}:", ln=1)
            for stat, value in stats.items():
                pdf.cell(0, 10, f"  {stat}: {value}", ln=1)
            pdf.ln(5)
        
        pdf.add_page()
        pdf.cell(0, 10, "Data Preview:", ln=1)
        for row in preview[:5]:
            row_text = ", ".join([f"{k}: {v}" for k, v in row.items()])
            pdf.multi_cell(0, 10, row_text)
            pdf.ln(2)
        
        # Add each chart to the PDF.
        import tempfile
        for chart_key, chart_base64 in charts.items():
            if chart_base64:
                img_data = base64.b64decode(chart_base64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(img_data)
                    tmp_filename = tmp_file.name
                pdf.add_page()
                pdf.cell(0, 10, chart_key.replace("_", " ").upper(), ln=1, align="C")
                pdf.image(tmp_filename, x=10, y=30, w=pdf.w - 20)
        
        pdf_bytes = pdf.output(dest="S").encode("latin1")
        return pdf_bytes, 200, {
            'Content-Type': 'application/pdf',
            'Content-Disposition': 'attachment; filename="report.pdf"'
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
