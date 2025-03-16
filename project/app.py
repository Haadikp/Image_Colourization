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


# 🔹 Step 1: Request Password Reset (Checks if email exists)
@app.route('/reset', methods=['GET', 'POST'])
def reset():
    if request.method == 'GET':
        return render_template('reset_password.html')

    # Clear any previous reset session data
    session.pop('reset_email', None)
    session.pop('otp', None)

    email = request.form.get('femail')
    if email:
        email = email.strip().lower()  # Validation: remove extra spaces and lowercase
    else:
        flash("Please enter your email address.","error")
        return redirect('/reset')

    # Check if email exists in the database
    userdata = execute_query('search', "SELECT * FROM user_login WHERE email = %s", (email,))
    print(userdata)

    if not userdata or userdata == ():
        # Display error message if user not found
        flash("Invalid email address! Please enter a registered email.","error")
        return redirect('/reset')  # Stay on the reset page

    # Generate OTP
    otp = random.randint(100000, 999999)

    # Store OTP & email in session
    session['reset_email'] = email
    session['otp'] = otp
    print(otp)
    # Send OTP via email
    msg = Message("Password Reset OTP", sender=app.config['MAIL_USERNAME'], recipients=[email])
    msg.body = f"Your OTP for password reset is: {otp}. This OTP will expire in 10 minutes."
    mail.send(msg)

    flash("An OTP has been sent to your email.","success")
    return redirect('/verify_otp')  # Move to OTP verification step


# 🔹 Step 2: OTP Verification (Checks if OTP is correct)
@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'GET':
        return render_template('verify_otp.html')  # Make sure this page exists

    email = session.get('reset_email')
    otp = session.get('otp')
    entered_otp = request.form.get('otp')
    print("entered otp:",entered_otp,"\nOtp sended:",otp)
    # If email is missing in session, restart reset process
    if not email:
        flash("Session expired! Please request password reset again.","error")
        return redirect('/reset')

    # Validate if email exists in database (ensuring security)
    userdata = execute_query('search', "SELECT * FROM user_login WHERE email = %s", (email,))
    if not userdata:
        flash("Invalid session! Please enter a valid email.","error")
        return redirect('/reset')

    # Check if entered OTP matches the stored one
    if entered_otp and int(entered_otp) == session.get('otp'):
        flash("OTP verified! Now enter your new password.", "success")
        return redirect('/set_new_password')  # Move to password reset step
    else:
        flash("Invalid OTP! Please enter the correct OTP.","error")
        return redirect('/verify_otp')  # Stay on OTP page
# 🔹 Step 3: Set New Password (Ensures OTP was verified)
@app.route('/set_new_password', methods=['GET', 'POST'])
def set_new_password():
    if request.method == 'GET':
        return render_template('set_password.html')  # Ensure this is correct

    email = session.get('reset_email')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    # If email is missing in session, restart reset process
    if not email:
        flash("Session expired! Please request password reset again.", "error")
        return redirect('/reset')

    # Check if passwords match
    if new_password != confirm_password:
        flash("Passwords do not match!", "error")
        return redirect('/set_new_password')

    # Validate email before updating password
    userdata = execute_query('search', "SELECT * FROM user_login WHERE email = %s", (email,))
    if not userdata:
        flash("Invalid session! Please enter a valid email.", "error")
        return redirect('/reset')

    # Hash the new password
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    # Update password in database
    try:
        query = "UPDATE user_login SET password = %s WHERE email = %s"
        execute_query('insert', query, (hashed_password, email))

        # Clear session
        session.pop('otp', None)
        session.pop('reset_email', None)

        flash("Your password has been reset successfully!", "success")
        return redirect('/')  # Redirect to login page
    except:
        flash("Something went wrong while resetting the password! Try again.", "error")
        return redirect('/reset')

# 🔹 Optional: Resend OTP Route
@app.route('/resend_otp', methods=['POST'])
def resend_otp():
    email = session.get('reset_email')

    # If no email stored in session, restart the reset process
    if not email:
        flash("Session expired! Please request password reset again.")
        return redirect('/reset')

    # Validate email in the database
    userdata = execute_query('search', "SELECT * FROM user_login WHERE email = %s", (email,))
    if not userdata:
        flash("Invalid session! Please enter a valid email.")
        return redirect('/reset')

    # Generate a new OTP
    otp = random.randint(100000, 999999)
    session['otp'] = otp  # Update OTP in session

    # Send new OTP email
    msg = Message("Resend OTP - Password Reset", sender=app.config['MAIL_USERNAME'], recipients=[email])
    msg.body = f"Your new OTP for password reset is: {otp}. This OTP will expire in 10 minutes."
    mail.send(msg)

    flash("A new OTP has been sent to your email.")
    return redirect('/verify_otp')








@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        flash("Already a user is logged-in!","error")
        return redirect('/')
    else:
        return render_template("register.html")


@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if 'user_id' not in session:
        name = request.form.get('name').strip()
        email = request.form.get('email').strip()
        passwd = request.form.get('password').strip()
        confirm_passwd = request.form.get('confirm-password').strip()

        if not name.replace(" ", "").isalpha() or len(name) < 5:
            flash("Name must be at least 5 characters long and contain only alphabetic characters.","error")
            return redirect('/register')

        email_regex = r'^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if not re.match(email_regex, email):
            flash("Invalid email format. Please enter a valid email address.","error")
            return redirect('/register')

        if len(passwd) < 5:
            flash("Password must be at least 5 characters long.","error")
            return redirect('/register')

        if passwd != confirm_passwd:
            flash("Passwords do not match!", "error")
            return redirect('/register')

        existing_user = execute_query('search', "SELECT * FROM user_login WHERE email = %s", (email,))
        if existing_user:
            flash("Email ID already exists, use another email!","error")
            return redirect('/register')

        try:
            hashed_password = bcrypt.hashpw(passwd.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            query = "INSERT INTO user_login(username, email, password) VALUES(%s, %s, %s)"
            execute_query('insert', query, (name, email, hashed_password))

            user = execute_query('search', "SELECT * FROM user_login WHERE email = %s", (email,))
            session['user_id'] = user[0][0]

            flash("Successfully Registered!", "success")
            return redirect('/')
        except Exception as e:
            flash(f"An error occurred during registration: {e}","error")
            return redirect('/register')
    else:
        flash("Already a user is logged-in!","error")
        return redirect('/')


# Load the pretrained GAN models
model_path1 = os.path.join(MODEL_FOLDER, "Main_Model.pth")
# model_path2 = os.path.join(MODEL_FOLDER, "Second_Model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load both models
net_G = torch.load(model_path1, map_location=device)
net_G.eval()  # Set first model to evaluation mode

# net_G2 = torch.load(model_path2, map_location=device)
# net_G2.eval()  # Set second model to evaluation mode

# Define transforms (if needed elsewhere)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# Preprocessing function

def preprocess_image(image_path, size=256):
    """
    Preprocess a grayscale image for the model.
    Returns a tensor for the L channel and the resized image.
    """
    img = Image.open(image_path).convert("RGB")  # Ensure RGB format
    transforms_pipeline = transforms.Compose([
        transforms.Resize((size, size), Image.BICUBIC)
    ])
    img_resized = transforms_pipeline(img)
    img_array = np.array(img_resized)
    img_lab = rgb2lab(img_array).astype("float32")
    L = img_lab[:, :, 0] / 50.0 - 1.0  # Normalize L to [-1,1]
    L_tensor = torch.tensor(L).unsqueeze(0).unsqueeze(0)  # Shape: (1,1,H,W)
    return L_tensor.to(device), img_resized


# Colorization function
def colorize_image(net_G, L_tensor):
    """
    Colorizes a grayscale image using the trained GAN model.
    Returns the final colorized image in RGB format.
    """
    with torch.no_grad():
        ab_pred = net_G.net_G(L_tensor)  # Generate predicted ab channels
    L = (L_tensor.squeeze().cpu().numpy() + 1.0) * 50.0  # Denormalize L
    ab = ab_pred.squeeze().cpu().numpy() * 110.0  # Denormalize ab
    ab = np.moveaxis(ab, 0, -1)  # Shape: (H, W, 2)
    lab_combined = np.zeros((L.shape[0], L.shape[1], 3))
    lab_combined[:, :, 0] = L
    lab_combined[:, :, 1:] = ab
    rgb_image = lab2rgb(lab_combined)
    return rgb_image


@app.route('/colorize', methods=['POST'])
def colorize_endpoint():
    """
    Receives an uploaded image, processes it, and returns the URL to the colorized image.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        L_tensor, original_image = preprocess_image(filepath)
        colorized_image = colorize_image(net_G, L_tensor)
        colorized_filename = f"colorized_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], colorized_filename)
        Image.fromarray((colorized_image * 255).astype(np.uint8)).save(result_path)
        result_url = url_for('result_file', filename=colorized_filename, _external=True)
        return jsonify({'colorized_image': result_url})
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500


# Flask routes
# @app.route('/home', methods=['GET', 'POST'])
# def index():
#     """
#     Homepage with upload functionality and displays the colorized image.
#     """
#     if request.method == 'GET' and 'colorized_image' not in session:
#         session.pop('colorized_image', None)  # Only clear when there's no processed image
#
#     colorized_filename = session.get('colorized_image', None)  # Use the correct session key
#
#     if request.method == 'POST':
#         # Check if the file is in the request
#         if 'file' not in request.files:
#             flash("No file uploaded!")
#             return redirect(request.url)
#
#         file = request.files['file']
#         if file.filename == '':
#             flash("No file selected!")
#             return redirect(request.url)
#
#         if file:
#             # Save the uploaded file
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(filepath)
#
#             # Redirect to process the uploaded image
#             return redirect(url_for('process_image', filename=file.filename))
#
#     return render_template('Home.html', colorized_image=colorized_filename)

@app.route('/home', methods=['GET', 'POST'])
def index():
    colorized_filename = session.get('colorized_image', None)
    return render_template('Home.html', colorized_image=colorized_filename)


@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Validate file size
        if len(file.read()) > 20 * 1024 * 1024:  # 20MB limit
            return jsonify({'error': 'File size exceeds 20MB limit'}), 400
        file.seek(0)  # Reset file pointer after reading

        # Validate file type
        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            return jsonify({'error': 'Unsupported file format'}), 400

        # Save original file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        return jsonify({
            'filename': filename,
            'preview_url': url_for('uploaded_file', filename=filename)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'webp'}


@app.route('/process/<filename>')
def process_image(filename):
    conversion_type = request.args.get('type', 'colorize')

    try:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(input_path):
            return jsonify({'error': 'File not found'}), 404

        # Generate unique output filename
        base_name = os.path.splitext(filename)[0]
        result_filename = f"{conversion_type}_{base_name}.png"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

        if conversion_type == 'grayscale':
            # Convert to grayscale
            img = Image.open(input_path).convert('L')
            img.save(result_path)
        elif conversion_type == 'colorize':
            # Colorize using GAN model
            L_tensor, _ = preprocess_image(input_path)
            with torch.no_grad():
                ab_pred = net_G.net_G(L_tensor)

            # Post-process and save
            colorized_image = postprocess_colorization(L_tensor, ab_pred)
            Image.fromarray(colorized_image).save(result_path)
        else:
            return jsonify({'error': 'Invalid conversion type'}), 400

        return jsonify({
            'result_url': url_for('result_file', filename=result_filename),
            'download_url': url_for('download_file', filename=result_filename)
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


def postprocess_colorization(L_tensor, ab_pred):
    """Convert model output to displayable RGB image"""
    L = L_tensor.cpu().squeeze().numpy()
    L = (L + 1) * 50  # Denormalize L channel
    ab = ab_pred.cpu().squeeze().numpy().transpose(1, 2, 0) * 110  # Denormalize ab

    lab = np.concatenate([L[..., np.newaxis], ab], axis=2)
    rgb = lab2rgb(lab) * 255
    return rgb.astype(np.uint8)


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(
        app.config['RESULT_FOLDER'],
        filename,
        as_attachment=True
    )
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

@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    flash('You have been logged out.', 'info')  # Optional: Display a logout message
    return redirect('/')  # Redirect to the login page

if __name__ == '__main__':
    app.run(debug=True)
