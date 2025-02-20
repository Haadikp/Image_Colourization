import bcrypt
import pymysql
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session
import os
from PIL import Image
import torch
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms
import re
import random
import bcrypt
from datetime import timedelta
from flask import session
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename

# file import MainModel,Unet,UnetBlock  # Adjust the import path to your actual `file.py`
from file import *  # Import all classes and functions from file.py
from flask import jsonify


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Folder paths
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'model'
RESULT_FOLDER = 'static/results'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Flask configurations
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER



# Database connection setup
db = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    database="image_colorization"
)

# Initialize Flask app, __name__ is passed to tell Flask the location of the app
# app = Flask(__name__)
# app.secret_key = os.urandom(24)  #secret key for the Flask app to secure session data.


# Helper function to execute queries
def execute_query(query_type, query, params=None):
    global cursor
    try:
        cursor = db.cursor()  # Create a cursor object
        cursor.execute(query, params or ())

        if query_type == "search":
            result = cursor.fetchall()
            cursor.close()
            return result
        elif query_type == "insert":
            db.commit()  # Commit the changes if it is an insert query
            cursor.close()
            return

    except pymysql.MySQLError as e:
        db.rollback()  # Rollback in case of error
        cursor.close()
        print(f"Database error: {e}")
        flash("An error occurred while processing your request.")
        # Handle or log the error as needed
        return None
    except Exception as e:
        cursor.close()
        print(f"Unexpected error: {e}")
        flash("An unexpected error occurred.")
        # Handle or log the error as needed
        return None


@app.route('/')
def login():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=25)
    if 'user_id' in session:
        flash("Already a user is logged-in!")
        return redirect('/home')
    else:
        return render_template("login.html")


@app.route('/login_validation', methods=['POST', 'GET'])
def login_validation():
    if 'user_id' not in session:
        email = request.form.get('email').strip()
        passwd = request.form.get('password').strip()
        query = "SELECT * FROM user_login WHERE email = %s"
        users = execute_query("search", query, (email,))

        if users:
            stored_password = users[0][3]  # Assuming the password is in the 4th column (hashed password)
            print(passwd.encode('utf-8'))
            print(stored_password.encode('utf-8'))
            if bcrypt.checkpw(passwd.encode('utf-8'), stored_password.encode('utf-8')):
                session['user_id'] = users[0][0]
                return redirect('/home')
            else:
                flash("Incorrect password. Please try again.")
                return redirect('/')
        else:
            flash("No account found with this email address.")
            return redirect('/')
    else:
        flash("Already a user is logged-in!")
        return redirect('/home')


# Flask-Mail configuration

app.config['SECRET_KEY'] = 'qwertyuiop'
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'kpgadgetsarena@gmail.com'
app.config['MAIL_PASSWORD'] = 'voxo isgt wxoi sqeb'
app.config['MAIL_DEBUG'] = True
mail = Mail(app)


# Password reset route using OTP
@app.route('/reset', methods=['POST'])
def reset():
    if 'user_id' not in session:
        email = request.form.get('femail')
        userdata = execute_query('search', f"SELECT * FROM user_login WHERE email = '{email}'")
        if userdata:
            # Generate a 6-digit OTP
            otp = random.randint(100000, 999999)

            # Store OTP and email in session for later validation
            session['reset_email'] = email
            session['otp'] = otp

            # Send OTP via email
            msg = Message("Password Reset OTP",
                          sender="noreply@app.com",
                          recipients=[email])
            msg.body = f"Your OTP for password reset is: {otp}. This OTP will expire in 10 minutes."
            mail.send(msg)

            flash("An OTP has been sent to your email.")
            return redirect('/verify_otp')  # Redirect to OTP verification page
        else:
            flash("Invalid email address!")
            return redirect('/')
    else:
        return redirect('/home')


# Route for verifying OTP and resetting password
@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        entered_otp = request.form.get('otp')
        new_password = request.form.get('new_password')

        if 'otp' in session and 'reset_email' in session:
            if int(entered_otp) == session['otp']:
                email = session['reset_email']

                # Hash the new password before updating
                hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

                try:
                    query = "UPDATE user_login SET password = %s WHERE email = %s"
                    execute_query('insert', query, (hashed_password, email))

                    session.pop('otp', None)
                    session.pop('reset_email', None)

                    flash("Your password has been reset successfully!")
                    return redirect('/')
                except:
                    flash("Something went wrong while resetting the password!")
                    return redirect('/verify_otp')
            else:
                flash('Invalid OTP. Please try again.')
                return redirect('/verify_otp')
        else:
            flash('Session expired or invalid. Please try again.')
            return redirect('/reset')

    return render_template('verify_otp.html')


@app.route('/register')
def register():
    if 'user_id' in session:
        flash("Already a user is logged-in!")
        return redirect('/home')
    else:
        return render_template("register.html")


@app.route('/registration', methods=['POST'])
def registration():
    if 'user_id' not in session:
        name = request.form.get('name').strip()
        email = request.form.get('email').strip()
        passwd = request.form.get('password').strip()

        if not name.replace(" ", "").isalpha() or len(name) < 5:
            flash("Name must be at least 5 characters long and contain only alphabetic characters.")
            return redirect('/register')

        email_regex = r'^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if not re.match(email_regex, email):
            flash("Invalid email format. Please enter a valid email address.")
            return redirect('/register')

        if len(passwd) < 5:
            flash("Password must be at least 5 characters long.")
            return redirect('/register')

        existing_user = execute_query('search', "SELECT * FROM user_login WHERE email = %s", (email,))
        if existing_user:
            flash("Email ID already exists, use another email!")
            return redirect('/register')

        try:
            hashed_password = bcrypt.hashpw(passwd.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            query = "INSERT INTO user_login(username, email, password) VALUES(%s, %s, %s)"
            execute_query('insert', query, (name, email, hashed_password))

            user = execute_query('search', "SELECT * FROM user_login WHERE email = %s", (email,))
            session['user_id'] = user[0][0]

            flash("Successfully Registered!")
            return redirect('/home')
        except Exception as e:
            flash(f"An error occurred during registration: {e}")
            return redirect('/register')
    else:
        flash("Already a user is logged-in!")
        return redirect('/home')


# Load the pretrained GAN model
model_path = os.path.join(MODEL_FOLDER, "Main_Model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load the GAN model
net_G = torch.load(model_path, map_location=device)
net_G.eval()  # Set the model to evaluation mode


# Preprocessing function
def preprocess_image(image_path, size=256):
    """
    Preprocess a grayscale image for the model.
    Args:
        image_path: Path to the grayscale image.
        size: Target size for the image (default: 256x256).
    Returns:
        Preprocessed L-channel tensor and the resized original grayscale image.
    """
    img = Image.open(image_path).convert("RGB")  # Ensure it's in RGB format
    transforms_pipeline = transforms.Compose([
        transforms.Resize((size, size), Image.BICUBIC)  # Resize to model's input size
    ])
    img_resized = transforms_pipeline(img)

    # Convert to LAB color space
    img_array = np.array(img_resized)
    img_lab = rgb2lab(img_array).astype("float32")  # Convert to LAB

    # Extract and normalize the L-channel
    L = img_lab[:, :, 0] / 50.0 - 1.0  # Normalize to [-1, 1]
    L_tensor = torch.tensor(L).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    return L_tensor.to(device), img_resized


# Colorization function
def colorize_image(net_G, L_tensor):
    """
    Colorize a grayscale image using the trained model.
    Args:
        model: Trained GAN model.
        L_tensor: Preprocessed L-channel tensor.
    Returns:
        Colorized image in RGB format.
    """
    with torch.no_grad():
        ab_pred = net_G.net_G(L_tensor)  # Generate predicted ab channels
    L = (L_tensor.squeeze().cpu().numpy() + 1.0) * 50.0  # Denormalize L
    ab = ab_pred.squeeze().cpu().numpy() * 110.0  # Denormalize ab
    ab = np.moveaxis(ab, 0, -1)  # Convert shape from (2, 256, 256) to (256, 256, 2)

    # Combine L and ab channels to form LAB and convert to RGB
    lab_combined = np.zeros((L.shape[0], L.shape[1], 3))
    lab_combined[:, :, 0] = L
    lab_combined[:, :, 1:] = ab
    rgb_image = lab2rgb(lab_combined)
    return rgb_image

@app.route('/colorize', methods=['POST'])
def colorize():
    """
    Receives an uploaded image, processes it, and returns the colorized image URL.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save and process the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    return redirect(url_for('process_image', filename=filename))


# Flask routes
@app.route('/home', methods=['GET', 'POST'])
def index():
    """
    Homepage with upload functionality and displays the colorized image.
    """
    if request.method == 'GET' and 'colorized_image' not in session:
        session.pop('colorized_image', None)  # Only clear when there's no processed image

    colorized_filename = session.get('colorized_image', None)  # Use the correct session key

    if request.method == 'POST':
        # Check if the file is in the request
        if 'file' not in request.files:
            flash("No file uploaded!")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("No file selected!")
            return redirect(request.url)

        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Redirect to process the uploaded image
            return redirect(url_for('process_image', filename=file.filename))

    return render_template('Home.html', colorized_image=colorized_filename)


@app.route('/process/<filename>', methods=['GET', 'POST'])
def process_image(filename):
    """
    Processes the uploaded image and performs colorization.
    """
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Check if file exists before processing
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400

        # Preprocess the image for the model
        L_tensor, original_image = preprocess_image(filepath)

        # Colorize the image using the trained model
        colorized_image = colorize_image(net_G, L_tensor)

        # Save the colorized image
        colorized_filename = f"colorized_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], colorized_filename)
        Image.fromarray((colorized_image * 255).astype(np.uint8)).save(result_path)

        # Update session with the new image
        session['colorized_image'] = colorized_filename

        # Return the correct path for the new image
        return jsonify({'colorized_image': url_for('result_file', filename=colorized_filename)})

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500




@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serves uploaded files for display.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/static/results/<filename>')
def result_file(filename):
    """
    Serves result files for display.
    """
    return send_from_directory(app.config['RESULT_FOLDER'], filename)



if __name__ == '__main__':
    app.run(debug=True)
