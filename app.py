import os
import io
import base64
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
from PIL import Image

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# IMPORTANT: Update this path to your actual model weights file
MODEL_PATH = 'model/best.pt'
# Use the SAME image size used during training
IMAGE_SIZE = 1408
CONFIDENCE_THRESHOLD = 0.25

# Class names (Ensure this order matches your data.yaml and training)
CLASS_NAMES = [
    'Broken_Root', 'PCT', 'Free_R_Max', 'Free_L_Max', 'Not_Free_Max',
    'Not_Free_Center_Max', 'Free_R_Mand', 'Free_L_Mand', 'Not_Free_Mand',
    'Not_Free_Center_Mand'
]
# ---------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key' # Needed for flashing messages

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load YOLO Model ---
try:
    print(f"Loading YOLOv8 model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"ERROR: Failed to load model from {MODEL_PATH}")
    print(f"Error details: {e}")
    print(f"Please ensure the path is correct and the model file is valid.")
    print(f"Exiting...")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    exit()
# ---------------------

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Diagnosis Function (Copied from your notebook) ---
def diagnosis(all_detected_classes):
    diagnosis_report=[]
    total_broken_roots=all_detected_classes.count('Broken_Root')
    if total_broken_roots==0: diagnosis_report.append('No broken root detected.')
    elif total_broken_roots==1: diagnosis_report.append('A single broken root detected.')
    else: diagnosis_report.append(f'{total_broken_roots} broken roots detected.')

    total_PCT=all_detected_classes.count('PCT')
    if total_PCT==0: diagnosis_report.append('No periodontally compromised tooth detected.')
    elif total_PCT==1: diagnosis_report.append('A single periodontally compromised tooth detected.')
    else: diagnosis_report.append(f'{total_PCT} periodontally compromised teeth detected.')

    max_findings = [cls for cls in all_detected_classes if 'Max' in cls]
    kennedy_max = "Maxillary Arch: Fully dentate or classification criteria not met."
    if max_findings:
        has_free_r_max = 'Free_R_Max' in max_findings
        has_free_l_max = 'Free_L_Max' in max_findings
        has_not_free_max = 'Not_Free_Max' in max_findings
        has_not_free_center_max = 'Not_Free_Center_Max' in max_findings
        count_not_free_max = max_findings.count('Not_Free_Max')
        if has_not_free_center_max: kennedy_max = 'Maxillary Kennedy classification: Class IV (Anterior edentulous area crossing midline)'
        elif has_free_r_max and has_free_l_max: kennedy_max = 'Maxillary Kennedy classification: Class I (Bilateral posterior edentulous areas)'
        elif has_free_r_max or has_free_l_max:
            side = "right" if has_free_r_max else "left"
            kennedy_max = f'Maxillary Kennedy classification: Class II (Unilateral posterior edentulous area - {side})'
        elif has_not_free_max:
            plural = "s" if count_not_free_max > 1 else ""
            kennedy_max = f'Maxillary Kennedy classification: Class III ({count_not_free_max} bounded edentulous area{plural})'
    diagnosis_report.append(kennedy_max)

    mand_findings = [cls for cls in all_detected_classes if 'Mand' in cls]
    kennedy_mand = "Mandibular Arch: Fully dentate or classification criteria not met."
    if mand_findings:
        has_free_r_mand = 'Free_R_Mand' in mand_findings
        has_free_l_mand = 'Free_L_Mand' in mand_findings
        has_not_free_mand = 'Not_Free_Mand' in mand_findings
        has_not_free_center_mand = 'Not_Free_Center_Mand' in mand_findings
        count_not_free_mand = mand_findings.count('Not_Free_Mand')
        if has_not_free_center_mand: kennedy_mand = 'Mandibular Kennedy classification: Class IV (Anterior edentulous area crossing midline)'
        elif has_free_r_mand and has_free_l_mand: kennedy_mand = 'Mandibular Kennedy classification: Class I (Bilateral posterior edentulous areas)'
        elif has_free_r_mand or has_free_l_mand:
            side = "right" if has_free_r_mand else "left"
            kennedy_mand = f'Mandibular Kennedy classification: Class II (Unilateral posterior edentulous area - {side})'
        elif has_not_free_mand:
            plural = "s" if count_not_free_mand > 1 else ""
            kennedy_mand = f'Mandibular Kennedy classification: Class III ({count_not_free_mand} bounded edentulous area{plural})'
    diagnosis_report.append(kennedy_mand)
    return diagnosis_report
# -------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                print(f"File saved to: {filepath}")

                # --- Run Inference ---
                print(f"Running inference on {filepath}...")
                results = model.predict(source=filepath,
                                        imgsz=IMAGE_SIZE,
                                        conf=CONFIDENCE_THRESHOLD,
                                        verbose=False) # Keep console clean
                print("Inference complete.")
                # --------------------

                annotated_image_data = None
                diagnosis_result = []
                raw_predictions = []
                detected_class_names = []

                if results and results[0]:
                    # Prepare raw predictions list for display
                    for i in range(len(results[0].boxes)):
                         label_idx = int(results[0].boxes.cls[i].item())
                         class_name = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else f"Unknown({label_idx})"
                         detected_class_names.append(class_name)
                         raw_predictions.append({
                             'label': label_idx,
                             'name': class_name,
                             'conf': results[0].boxes.conf[i].item(),
                             'box': results[0].boxes.xywhn[i].tolist() # Normalized xywh
                         })

                    # Get diagnosis
                    diagnosis_result = diagnosis(detected_class_names)

                    # Generate annotated image for display
                    try:
                        annotated_img_bgr = results[0].plot() # Returns BGR numpy array
                        annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(annotated_img_rgb)

                        # Convert PIL Image to Base64 string
                        buffer = io.BytesIO()
                        img_pil.save(buffer, format="PNG") # Save as PNG bytes
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        annotated_image_data = f"data:image/png;base64,{img_base64}"
                        print("Annotated image generated.")
                    except Exception as e:
                        print(f"Error plotting/encoding annotated image: {e}")
                        flash(f"Error generating annotated image: {e}")

                else:
                     print("No results found or prediction failed.")
                     diagnosis_result = ["No objects detected above threshold."]


                # Clean up uploaded file after processing
                try:
                    os.remove(filepath)
                    print(f"Removed temporary file: {filepath}")
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {filepath}: {e}")


                # Render the results page
                return render_template('results.html',
                                       filename=filename,
                                       diagnosis=diagnosis_result,
                                       predictions=raw_predictions,
                                       annotated_image_data=annotated_image_data)

            except Exception as e:
                 print(f"Error during processing file {filename}: {e}")
                 flash(f"An error occurred processing the file: {e}")
                 # Clean up if error occurred after saving
                 if os.path.exists(filepath):
                     try:
                         os.remove(filepath)
                     except Exception as rm_err:
                         print(f"Warning: Could not remove file {filepath} after error: {rm_err}")
                 return redirect(request.url)

        else:
            flash('Invalid file type. Allowed types: png, jpg, jpeg')
            return redirect(request.url)

    # Render the upload form for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask server...")
    # host='0.0.0.0' makes it accessible on your local network
    # Use host='127.0.0.1' (or remove host=...) for access only on your machine
    app.run(debug=True, host='0.0.0.0', port=5000)