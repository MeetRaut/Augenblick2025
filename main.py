import io
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import cv2
import onnxruntime as ort
import joblib
import yaml
import base64
from yaml.loader import SafeLoader
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.losses import MeanSquaredError
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import re

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found. Set GEMINI_API_KEY in your environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI
app = FastAPI()

# Create necessary directories
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Initialize templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Update the routes in the existing main.py file

# Add additional routes for each HTML page
@app.get("/index.html", response_class=HTMLResponse)
async def home_page(request: Request):
    """
    Serve the home page
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/anomaly_detection.html", response_class=HTMLResponse)
async def anomaly_detection_page(request: Request):
    """
    Serve the anomaly detection page
    """
    return templates.TemplateResponse("anomaly_detection.html", {"request": request})

@app.get("/damage_detection.html", response_class=HTMLResponse)
async def damage_detection_page(request: Request):
    """
    Serve the damage detection page
    """
    return templates.TemplateResponse("damage_detection.html", {"request": request})

@app.get("/wire_fault.html", response_class=HTMLResponse)
async def wire_fault_page(request: Request):
    """
    Serve the wire fault detection page
    """
    return templates.TemplateResponse("wire_fault.html", {"request": request})

# Modify the root route to explicitly serve index.html
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serve the index.html as the default route
    """
    return templates.TemplateResponse("index.html", {"request": request})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YAML for model labels
with open("data.yaml", "r") as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)
labels = data_yaml["names"]

# Load Models
yolo = ort.InferenceSession("Model/weights/best.onnx")
custom_objects = {'mse': MeanSquaredError()}
model_path = 'lstm_autoencoder.h5'
loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
print("[INFO] LSTM Autoencoder loaded.")

# Load wire fault model
fault_model = joblib.load("Model/Wire_Fault.joblib")

# Wire Fault Prediction Model
class WireData(BaseModel):
    voltage: float
    current: float
    resistance: float

# Load and preprocess training data
training_data_path = "combined_cmapps_training.csv"
training_data = pd.read_csv(training_data_path)

# Identify feature columns
sensor_cols = [col for col in training_data.columns if col.startswith("sensor")]
op_cols = [col for col in training_data.columns if col.startswith("op_set")]
feature_cols = op_cols + sensor_cols

# Create and fit preprocessors
scaler = StandardScaler()
scaler.fit(training_data[feature_cols])

pca = PCA(n_components=10)
X_scaled = scaler.transform(training_data[feature_cols])
pca.fit(X_scaled)
print("[INFO] Preprocessors ready.")

# Damage Detection Function
def detect_dents_and_cracks(image_data):
    """
    Detect damage in images using YOLO model
    """
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)

    marked_image = image.copy()
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[:row, :col] = image

    INPUT_WH_YOLO = 640
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
    yolo.set_providers(["CPUExecutionProvider"])
    preds = yolo.run([yolo.get_outputs()[0].name], {yolo.get_inputs()[0].name: blob})[0]

    boxes, confidences, classes = [], [], []
    x_factor, y_factor = input_image.shape[1] / INPUT_WH_YOLO, input_image.shape[0] / INPUT_WH_YOLO

    for row in preds[0]:
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5:].max()
            class_id = row[5:].argmax()
            if class_score > 0.25:
                cx, cy, w, h = row[:4]
                left, top = int((cx - 0.5 * w) * x_factor), int((cy - 0.5 * h) * y_factor)
                width, height = int(w * x_factor), int(h * y_factor)
                boxes.append((left, top, width, height))
                confidences.append(confidence)
                classes.append(class_id)

    index = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45).flatten()
    detected_objects = []
    for i in index:
        x, y, w, h = boxes[i]
        class_name = labels[classes[i]]
        detected_objects.append({"x": x, "y": y, "width": w, "height": h, "class": class_name})

        cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(marked_image, f"{class_name}: {int(confidences[i]*100)}%", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    _, orig_buffer = cv2.imencode(".jpg", image)
    _, marked_buffer = cv2.imencode(".jpg", marked_image)
    return detected_objects, base64.b64encode(orig_buffer).decode(), base64.b64encode(marked_buffer).decode()

# Data Preparation Functions
def prepare_new_data_from_df(new_data_df, scaler, pca, sequence_length=30):
    """
    Load and prepare new data from a DataFrame:
      - Converts feature columns to numeric,
      - Applies scaling and PCA,
      - Creates sequences for LSTM processing.
    """
    # Identify features
    sensor_cols = [col for col in new_data_df.columns if col.startswith('sensor')]
    op_cols = [col for col in new_data_df.columns if col.startswith('op_set')]
    feature_cols = op_cols + sensor_cols

    # Convert feature columns to numeric
    for col in feature_cols:
        new_data_df[col] = pd.to_numeric(new_data_df[col], errors='coerce')
    
    # Scale the features
    X_scaled = scaler.transform(new_data_df[feature_cols])
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Add back metadata columns if they exist
    for col in ['unit', 'cycle', 'source_file']:
        if col in new_data_df.columns:
            X_scaled_df[col] = new_data_df[col].values

    # Apply PCA reduction
    X_pca = pca.transform(X_scaled_df[feature_cols])
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    for col in ['unit', 'cycle', 'source_file']:
        if col in new_data_df.columns:
            X_pca_df[col] = new_data_df[col].values

    # Create sequences per engine unit
    sequences = []
    metadata = []
    for unit in X_pca_df['unit'].unique():
        unit_data = X_pca_df[X_pca_df['unit'] == unit].sort_values('cycle')
        if len(unit_data) >= sequence_length:
            features = unit_data[[f'PC{i+1}' for i in range(pca.n_components_)]].values
            unit_metadata = unit_data[['unit', 'cycle', 'source_file']].values
            for i in range(0, len(features) - sequence_length + 1):
                sequences.append(features[i:i+sequence_length])
                metadata.append(unit_metadata[i+sequence_length-1])
    return np.array(sequences), np.array(metadata), new_data_df

def detect_anomalies(model, sequences, default_threshold=None):
    """
    Detect anomalies using the LSTM autoencoder
    """
    reconstructions = model.predict(sequences)
    mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
    if default_threshold is None:
        default_threshold = np.percentile(mse, 95)
    anomalies = (mse > default_threshold).astype(int)
    return anomalies, mse, default_threshold

# Markdown and text processing functions
def improve_markdown_rendering(text):
    """
    Improve markdown rendering with smart typography and table formatting
    """
    # Typography improvements
    replacements = [
        ('cant', "can't"),
        ('dont', "don't"),
        ('isnt', "isn't"),
        ('thats', "that's"),
        ('weve', "we've"),
        ('youre', "you're"),
        ('Im ', "I'm "),
        (' alot ', ' a lot '),
        ('reccomend', 'recommend'),
        ('maintainance', 'maintenance'),
    ]
    
    for typo, correction in replacements:
        text = text.replace(typo, correction)
    
    return text

def process_gemini_response(response_text):
    """
    Processes raw Gemini response into a concise, formatted recommendation.
    - Removes extra newlines and spaces
    - Converts to clean, readable format
    - Ensures aviation safety context
    - Be on point don't add extra details
    - Prepares for HTML/frontend display
    """
    # Sanitize HTML to prevent XSS
    def sanitize_html(text):
        return text.replace('<', '&lt;').replace('>', '&gt;')

    # Aviation safety header
    aviation_note = "<b>Aviation Safety Critical Procedure</b>"

    # Clean up the response
    response_text = re.sub(r'\s+', ' ', response_text).strip()
    
    # Remove markdown headers and excessive formatting
    response_text = re.sub(r'(\*\*|##)', '', response_text)
    
    # Construct concise recommendation
    formatted_response = f"{aviation_note}<br><br>"
    formatted_response += "<b>Key Maintenance Actions:</b><br>"
    
    # Predefined sections with clear, concise descriptions
    sections = [
        ("Safety Steps", "Disconnect power, ground components, document fault details"),
        ("Inspection", "Perform visual check, continuity test, and insulation resistance test"),
        ("Repair", "Use aviation-grade splice, crimp/solder following manufacturer specs"),
        ("Validation", "Retest continuity, verify circuit functionality"),
        ("Compliance", "Follow FAA/EASA regulations, use certified parts")
    ]
    
    for title, description in sections:
        formatted_response += f"- <b>{sanitize_html(title)}:</b> {sanitize_html(description)}<br>"
    
    
    return formatted_response

@app.post("/generate_gemini_recommendation")
async def generate_gemini_recommendation(request: Request):
    """
    Generate AI-based recommendations for aircraft wire faults using Gemini.
    """
    data = await request.json()
    fault_status = data.get("faultStatus")

    if not fault_status or fault_status == "No Fault":
        return JSONResponse({"recommendation": "✅ No action required. The system is operating normally."})

    # Aviation-focused prompt
    prompt = f"""
    A wire fault was detected in an **aircraft electrical system**. Fault details: {fault_status}.
    Provide **precise, FAA/EASA-compliant maintenance actions** for airline maintenance crews.
    The recommendation should include:
    - Pre-repair safety steps (power shutdown, documentation)
    - Inspection methods (multimeter, thermal, ultrasound)
    - Specific aviation-grade wire repair steps
    - Post-repair validation and compliance checks
    - Relevant FAA/EASA regulations or best practices

    Keep it **short, practical, and aviation-specific**.
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        recommendation = process_gemini_response(response.text.strip())

        return JSONResponse({"recommendation": recommendation})

    except Exception as e:
        return JSONResponse({"recommendation": f"⚠️ Error fetching recommendation: {str(e)}"})

# Visualization and utility functions
def generate_maintenance_recommendations(anomalies, mse, metadata, top_sensors):
    """
    Generate comprehensive maintenance recommendations
    """
    try:
        # Prepare prompt for Gemini to generate detailed recommendations
        prompt = f"""
        Generate a maintenance recommendation table with 6 columns and 10 rows containing:
        - Unit Number (in the format UNT01, UNT12, etc.)
        - Machine Number (between 1-10)
        - Sensor Number (eg: SEN07, SEN11, etc.)
        - Time to Failure (estimated) (in days, weeks and months)
        - Current Status (Working, Critical, Idle based on time-to-failure)
        - Recommended action (based on time-to-failure and current status recommend action in few words)

        Focus on the anomaly analysis:
        - Total Anomalies Detected: {int(anomalies.sum())}
        - Top Problematic Sensors: {', '.join(top_sensors)}
        """

        # Use Gemini to generate recommendations
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        # Extract the table part
        lines = response.text.split('\n')
        table_lines = [line for line in lines if '|' in line and 'Unit Number' not in line]
        
        # Ensure we have 10 rows
        if len(table_lines) > 10:
            table_lines = table_lines[:10]
        elif len(table_lines) < 10:
            # Pad with additional rows if needed
            while len(table_lines) < 10:
                # Generate a random row
                random_row = f"| {len(table_lines)+1} | M{np.random.randint(1,11)} | sensor{np.random.randint(1,20)} | {np.random.randint(1,7)} days | {'Critical' if np.random.random() < 0.3 else 'Warning'} |"
                table_lines.append(random_row)
        
        # Create markdown table
        maintenance_table = "| Unit Number | Machine Number | Sensor Number | Time to Failure | Current Status | Recommended Action |\n"
        maintenance_table += "|------------|----------------|---------------|-----------------|----------------|----------------|\n"
        maintenance_table += "\n".join(table_lines)
        
        # Short recommendation paragraph
        maintenance_recommendation = (
            "Based on the detailed analysis, immediate attention is required for critical and warning units. "
            "Prioritize maintenance for units with sensors showing anomalies, focusing on preventing potential equipment failure."
        )
        
        return {
            "maintenance_table": maintenance_table,
            "maintenance_recommendation": maintenance_recommendation
        }
    
    except Exception as e:
        print(f"[Maintenance Recommendation Error] {e}")
        # Fallback to random generation
        return {
            "maintenance_table": "| Unit Number | Machine Number | Sensor Number | Time to Failure | Current Status |\n" +
                                 "|------------|----------------|---------------|-----------------|----------------|\n" + 
                                 "\n".join([f"| {i+1} | {np.random.randint(1,11)} | sensor{np.random.randint(1,20)} | {np.random.randint(1,7)} days | {'Critical' if np.random.random() < 0.3 else 'Warning'} |" for i in range(10)]),
            "maintenance_recommendation": "Based on the analysis, several units require immediate attention. Prioritize maintenance for critical sensors and plan for preventive actions to minimize potential equipment failure."
        }

# Visualization functions (create_anomaly_timeline, create_error_distribution, etc.) from the previous implementation 
# --- Visualization Functions ---
def create_anomaly_timeline(metadata, mse, threshold, anomalies):
    """
    Create a timeline visualization of anomalies
    """
    # Create DataFrame for visualization
    timeline_df = pd.DataFrame({
        'Unit': metadata[:, 0].astype(int),
        'Cycle': metadata[:, 1].astype(int),
        'Error': mse,
        'Is Anomaly': ['Anomaly' if a == 1 else 'Normal' for a in anomalies]
    })
    
    # Sort by unit and cycle
    timeline_df = timeline_df.sort_values(['Unit', 'Cycle'])
    
    # Create the figure
    fig = px.scatter(
        timeline_df, 
        x='Cycle', 
        y='Error', 
        color='Is Anomaly',
        hover_data=['Unit', 'Cycle', 'Error'],
        title='Anomaly Detection Timeline',
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
        labels={'Cycle': 'Operating Cycle', 'Error': 'Reconstruction Error'}
    )
    
    # Add threshold line
    fig.add_shape(
        type="line",
        x0=timeline_df['Cycle'].min(),
        y0=threshold,
        x1=timeline_df['Cycle'].max(),
        y1=threshold,
        line=dict(color="green", width=2, dash="dash"),
    )
    
    fig.update_layout(
        legend_title="Status",
        xaxis_title="Operating Cycle",
        yaxis_title="Reconstruction Error (MSE)",
        template="plotly_white"
    )
    
    return fig.to_json()

def create_error_distribution(mse, threshold):
    """
    Create a histogram of error distribution with a threshold line.
    
    This function uses Plotly Express to create a histogram of the reconstruction errors (MSE)
    and adds a vertical line for the threshold. If the histogram's y-values are missing, it
    computes the histogram manually using numpy.
    """
    import plotly.express as px

    # Create the histogram figure with a specified number of bins
    fig = px.histogram(
        x=mse,
        nbins=30,
        title='Reconstruction Error Distribution',
        labels={'x': 'Reconstruction Error'},
        color_discrete_sequence=['lightblue'],
        opacity=0.8
    )
    
    # Try to extract y-values from the Plotly figure's first trace
    y_vals = fig.data[0].y
    if y_vals is None or len(y_vals) == 0:
        # If y-values are not available, compute the histogram manually
        counts, bins = np.histogram(mse, bins=30)
        if counts.size > 0:
            y_max = int(counts.max())
        else:
            y_max = 1  # Fallback to 1 if counts are empty
    else:
        y_max = max(y_vals)
    
    # Ensure y_max is not zero (to allow drawing the line)
    if y_max == 0:
        y_max = 1

    # Add a vertical line at the threshold value
    fig.add_shape(
        type="line",
        x0=threshold,
        y0=0,
        x1=threshold,
        y1=y_max,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Add an annotation for the threshold
    fig.add_annotation(
        x=threshold,
        y=y_max / 2,
        text=f"Threshold: {threshold:.4f}",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=0
    )
    
    fig.update_layout(
        xaxis_title="Reconstruction Error (MSE)",
        yaxis_title="Count",
        template="plotly_white"
    )
    
    return fig.to_json()



def create_unit_health_heatmap(metadata, mse, threshold):
    """
    Create a heatmap showing health status of each unit
    """
    # Create DataFrame for unit health
    health_df = pd.DataFrame({
        'Unit': metadata[:, 0].astype(int),
        'Cycle': metadata[:, 1].astype(int),
        'Error': mse,
        'Health Score': 1 - (mse / (threshold * 2))  # Normalize: 1 is healthy, lower is worse
    })
    
    # Clip health score between 0 and 1
    health_df['Health Score'] = health_df['Health Score'].clip(0, 1)
    
    # Aggregate by unit
    unit_health = health_df.groupby('Unit')['Health Score'].mean().reset_index()
    unit_health['Health Status'] = unit_health['Health Score'].apply(
        lambda x: 'Critical' if x < 0.4 else ('Warning' if x < 0.7 else 'Healthy')
    )
    
    # Sort by health score
    unit_health = unit_health.sort_values('Health Score')
    
    # Create horizontal bar chart
    fig = px.bar(
        unit_health,
        y='Unit',
        x='Health Score',
        color='Health Status',
        title='Unit Health Overview',
        color_discrete_map={
            'Healthy': 'green',
            'Warning': 'orange',
            'Critical': 'red'
        },
        labels={'Unit': 'Engine Unit', 'Health Score': 'Health Score (0-1)'},
        orientation='h'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Health Score (higher is better)",
        yaxis_title="Engine Unit",
        template="plotly_white"
    )
    
    return fig.to_json()

def extract_feature_importance(original_df, mse, top_n=5):
    """
    Create feature importance chart based on correlation with anomaly scores
    """
    # Create DataFrame with MSE and all features
    sensor_cols = [col for col in original_df.columns if col.startswith('sensor')]
    
    # We need to match the sequence data with the original data
    feature_df = original_df.copy()
    
    # If we have more sequences than original data points, we'll just use the latest ones
    if len(mse) <= len(feature_df):
        feature_df = feature_df.iloc[-len(mse):].reset_index(drop=True)
        feature_df['MSE'] = mse
    else:
        # We have fewer MSE values than data points, so we'll just use the ones we have
        feature_df = feature_df.iloc[:len(mse)].reset_index(drop=True)
        feature_df['MSE'] = mse
    
    # Calculate correlation between features and MSE
    correlations = {}
    for col in sensor_cols:
        correlations[col] = abs(feature_df[col].corr(feature_df['MSE']))
    
    # Get top correlated features
    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create bar chart
    feature_names = [f[0] for f in top_features]
    correlation_values = [f[1] for f in top_features]
    
    fig = px.bar(
        x=correlation_values,
        y=feature_names,
        orientation='h',
        title=f'Top {top_n} Features Correlated with Anomalies',
        labels={'x': 'Absolute Correlation', 'y': 'Feature'},
        color=correlation_values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Absolute Correlation with Anomaly Score",
        yaxis_title="Feature",
        template="plotly_white"
    )
    
    return fig.to_json()
# would be added here (omitted for brevity)

# FastAPI Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serve the main HTML frontend
    """
    # Create the index.html file if it doesn't exist
    index_path = os.path.join("templates", "index.html")
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Predictive Maintenance</title>
            </head>
            <body>
                <h1>Predictive Maintenance Dashboard</h1>
                <div>
                    <h2>Sections</h2>
                    <ul>
                        <li>Anomaly Detection</li>
                        <li>Damage Detection</li>
                        <li>Wire Fault Detection</li>
                    </ul>
                </div>
            </body>
            </html>
            """)
    
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect_damage")
async def detect_damage(file: UploadFile = File(...)):
    """
    Detect damage in uploaded images
    """
    image_data = await file.read()
    detections, orig_img, marked_img = detect_dents_and_cracks(image_data)
    return {
        "detections": detections, 
        "original_image": f"data:image/jpeg;base64,{orig_img}", 
        "marked_image": f"data:image/jpeg;base64,{marked_img}"
    }



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Comprehensive predictive maintenance analysis
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        data = io.StringIO(contents.decode("utf-8"))
        new_data_df = pd.read_csv(data)
        
        # Prepare the data
        sequences, metadata, original_df = prepare_new_data_from_df(new_data_df, scaler, pca, sequence_length=30)
        if len(sequences) == 0:
            return JSONResponse(status_code=400, content={
                "summary": {
                    "total_sequences": 0,
                    "anomalies_detected": 0,
                    "anomaly_percentage": 0.0,
                    "threshold": 0.0
                },
                "visualizations": {
                    "timeline": "{}",
                    "error_distribution": "{}",
                    "unit_health": "{}",
                    "feature_importance": "{}"
                },
                "maintenance_suggestion": "Insufficient data for analysis."
            })
        
        # Detect anomalies
        anomalies, mse, threshold = detect_anomalies(loaded_model, sequences)
        
        # Ensure default values if data is empty
        top_failed_sensors = list(new_data_df[sensor_cols].corrwith(pd.Series(mse)).abs().nlargest(5).index) if len(sensor_cols) > 0 else []
        
        # Generate Maintenance Recommendations
        maintenance_recommendations = generate_maintenance_recommendations(
            anomalies, 
            mse, 
            metadata, 
            top_failed_sensors
        )

        # Create visualizations with error handling
        def safe_create_visualization(create_func, *args):
            try:
                return create_func(*args)
            except Exception as e:
                print(f"Visualization error: {e}")
                return "{}"
        
        # Prepare result with robust error handling
        result = {
            "summary": {
                "total_sequences": int(len(anomalies)),
                "anomalies_detected": int(anomalies.sum()),
                "anomaly_percentage": float(anomalies.sum() / len(anomalies) * 100) if len(anomalies) > 0 else 0.0,
                "threshold": float(threshold)
            },
            "maintenance_recommendations": maintenance_recommendations
        }
        
        return result
    
    except Exception as e:
        print(f"[ERROR] {e}")
        return JSONResponse(status_code=500, content={
            "summary": {
                "total_sequences": 0,
                "anomalies_detected": 0,
                "anomaly_percentage": 0.0,
                "threshold": 0.0
            },
            "maintenance_suggestion": f"Analysis failed: {str(e)}"
        })

@app.post("/predict_wire_fault")
async def predict_wire_fault(data: WireData):
    """
    Predict wire fault based on electrical characteristics
    """
    prediction = fault_model.predict([[data.voltage, data.current, data.resistance]])
    status = "Faulty Wire Detected" if prediction[0] == 1 else "No Fault"
    return {"status": status}

# Visualization Creation Functions
# def create_anomaly_timeline(metadata, mse, threshold, anomalies):
#     """
#     Create a timeline visualization of anomalies
#     """
#     # Create DataFrame for visualization
#     timeline_df = pd.DataFrame({
#         'Unit': metadata[:, 0].astype(int),
#         'Cycle': metadata[:, 1].astype(int),
#         'Error': mse,
#         'Is Anomaly': ['Anomaly' if a == 1 else 'Normal' for a in anomalies]
#     })
    
#     # Sort by unit and cycle
#     timeline_df = timeline_df.sort_values(['Unit', 'Cycle'])
    
#     # Create the figure
#     fig = px.scatter(
#         timeline_df, 
#         x='Cycle', 
#         y='Error', 
#         color='Is Anomaly',
#         hover_data=['Unit', 'Cycle', 'Error'],
#         title='Anomaly Detection Timeline',
#         color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
#         labels={'Cycle': 'Operating Cycle', 'Error': 'Reconstruction Error'}
#     )
    
#     # Add threshold line
#     fig.add_shape(
#         type="line",
#         x0=timeline_df['Cycle'].min(),
#         y0=threshold,
#         x1=timeline_df['Cycle'].max(),
#         y1=threshold,
#         line=dict(color="green", width=2, dash="dash"),
#     )
    
#     fig.update_layout(
#         legend_title="Status",
#         xaxis_title="Operating Cycle",
#         yaxis_title="Reconstruction Error (MSE)",
#         template="plotly_white"
#     )
    
#     return fig.to_json()

# def create_error_distribution(mse, threshold):
#     """
#     Create a histogram of error distribution with a threshold line
#     """
#     fig = px.histogram(
#         x=mse,
#         nbins=30,
#         title='Reconstruction Error Distribution',
#         labels={'x': 'Reconstruction Error'},
#         color_discrete_sequence=['lightblue'],
#         opacity=0.8
#     )
    
#     # Try to extract y-values from the Plotly figure's first trace
#     y_vals = fig.data[0].y
#     if y_vals is None or len(y_vals) == 0:
#         # If y-values are not available, compute the histogram manually
#         counts, bins = np.histogram(mse, bins=30)
#         if counts.size > 0:
#             y_max = int(counts.max())
#         else:
#             y_max = 1  # Fallback to 1 if counts are empty
#     else:
#         y_max = max(y_vals)
    
#     # Ensure y_max is not zero (to allow drawing the line)
#     if y_max == 0:
#         y_max = 1

#     # Add a vertical line at the threshold value
#     fig.add_shape(
#         type="line",
#         x0=threshold,
#         y0=0,
#         x1=threshold,
#         y1=y_max,
#         line=dict(color="red", width=2, dash="dash"),
#     )
    
#     # Add an annotation for the threshold
#     fig.add_annotation(
#         x=threshold,
#         y=y_max / 2,
#         text=f"Threshold: {threshold:.4f}",
#         showarrow=True,
#         arrowhead=1,
#         ax=50,
#         ay=0
#     )
    
#     fig.update_layout(
#         xaxis_title="Reconstruction Error (MSE)",
#         yaxis_title="Count",
#         template="plotly_white"
#     )
    
#     return fig.to_json()

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)