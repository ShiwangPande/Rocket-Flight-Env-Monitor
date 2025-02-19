# Rocket-Flight-Env-Monitor

This project provides a backend service for analyzing environmental data collected during rocket flights.  The system ingests CSV data, generates summary statistics, creates visualizations, and outputs a comprehensive PDF report.

## Features and Functionality

* **CSV Upload and Processing:** Accepts CSV files containing environmental sensor data.
* **Data Validation:** Checks for the presence of essential columns specified in `backend/app.py` (`IMPORTANT_COLS`).
* **Data Summary:** Calculates descriptive statistics (count, mean, std, etc.) for key parameters.
* **Data Visualization:** Generates various charts (line charts, scatter plots) illustrating trends in temperature, humidity, pressure, altitude, acceleration, and air quality.
* **Air Quality Index (AQI) Calculation:** Computes AQI based on PM2.5 concentration using US EPA breakpoints.
* **PDF Report Generation:** Creates a PDF report including summary statistics, data preview, and generated charts.


## Technology Stack

* **Backend:** Python (Flask framework)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib
* **PDF Generation:** FPDF
* **Data Serialization:** JSON

## Prerequisites

* Python 3.7 or higher
* Required Python packages: `Flask`, `pandas`, `numpy`, `matplotlib`, `flask-cors`, `fpdf`
  You can install these using pip:
  ```bash
  pip install Flask pandas numpy matplotlib flask-cors fpdf
  ```

## Installation Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/ShiwangPande/Rocket-Flight-Env-Monitor.git
   ```
2. Navigate to the backend directory:
   ```bash
   cd Rocket-Flight-Env-Monitor/backend
   ```
3. Install the required Python packages (see Prerequisites).
4. Run the Flask application:
   ```bash
   python app.py
   ```
   The application will start on port 5001.  Ensure you have the port open if using a firewall.


## Usage Guide

1. **Upload CSV Data:** Use a tool like `curl` or Postman to send a POST request to `/api/upload` with the CSV file as a multipart form-data file named 'file'.

   Example using `curl`:
   ```bash
   curl -X POST -F "file=@path/to/your/data.csv" http://localhost:5001/api/upload
   ```

2. **Download PDF Report:**  Use a tool like `curl` or Postman to send a POST request to `/api/download` with the same CSV file.  The response will be a PDF file.

    Example using `curl`:
    ```bash
    curl -X POST -F "file=@path/to/your/data.csv" http://localhost:5001/api/download > report.pdf
    ```

   The response will contain the JSON analysis including summary statistics, data preview, and base64 encoded charts, including error handling.  The `/api/download` endpoint will return a PDF directly.



## API Documentation

**Endpoint:** `/api/upload` (POST)

* **Request:**  `multipart/form-data`, including a file named `file` (CSV data).
* **Response (Success - 200 OK):** JSON object containing:
    * `message`: "File processed successfully"
    * `basic_summary`: Summary statistics of the data.
    * `data_preview`:  First 10 rows of the data.
    * `charts`: Base64 encoded images of the generated charts.
* **Response (Error - 400 Bad Request or 500 Internal Server Error):** JSON object containing an `error` message.

**Endpoint:** `/api/download` (POST)

* **Request:**  `multipart/form-data`, including a file named `file` (CSV data).
* **Response (Success - 200 OK):** PDF report as a binary file.
* **Response (Error - 400 Bad Request or 500 Internal Server Error):** JSON object containing an `error` message.


## Contributing Guidelines

Contributions are welcome! Please open issues or submit pull requests on GitHub.  Adhere to PEP 8 style guidelines for Python code.


## License Information

(Not specified in the repository.  Please add a license.)


## Contact/Support Information

(Please add contact information.)
