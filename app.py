import os
import cv2
from flask import Flask, render_template, request, redirect
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from keras.models import load_model
import joblib
from flask_security import Security, login_required, SQLAlchemySessionUserDatastore
from database import db_session, init_db
from models import User, Role
from flask_security.utils import encrypt_password, logout_user
from flask_mail import Mail

UPLOAD_FOLDER = 'user_uploads/'
ALLOWED_EXTENSIONS = ('png', 'jpg', 'jpeg')
pred_class_name = ""

def check_or_make_folder(foldername):
    if not os.path.exists(foldername):
        os.mkdir(foldername)

mail = Mail()

check_or_make_folder(UPLOAD_FOLDER)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'errnasuppsespcreerrnasecrernaret'
app.config['SECURITY_PASSWORD_HASH'] = 'plaintext'
app.config['SECURITY_REGISTERABLE'] = True
app.config['SECURITY_SEND_REGISTER_EMAIL'] = False
model = load_model('saved_models/current_model.h5')
mapper = joblib.load('saved_models/current_model_mapping.pkl')
mapper = {v: k for k, v in mapper.items()}

mail.init_app(app)

# Setup Flask-Security
user_datastore = SQLAlchemySessionUserDatastore(db_session, User, Role)
security = Security(app, user_datastore)

# Create a user to test with
@app.before_first_request
def create_user():
    if os.path.exists('test.db'):
        os.remove('test.db')
    init_db()
    user_datastore.create_user(email='test@example.com', password=encrypt_password('password'))
    db_session.commit()

def allowed_file(filename): 
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/")
@login_required
def main():
    return render_template('home.html')

@app.route("/logout", methods=['GET'])
@login_required
def logout_user():
    logout_user()

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        pic = preprocess_single_image(filepath)
        pred_probs = model.predict(pic)
        pred_class = pred_probs.argmax(axis=-1)[0]
        global pred_class_name 
        pred_class_name = get_pred_class_name(pred_class)
        print(get_pred_class_name(pred_class))
        
        return render_template('display.html', dic={})

    return "Upload rejected"

@app.route("/display")
@login_required
def test():
    global pred_class_name
    dic = {"pred_class_name": pred_class_name}
    return render_template('display.html', dic=dic)

@app.route('/login')
def login():
    return 'Login'

def get_pred_class_name(pred_class_number):
    global mapper
    return mapper[pred_class_number]

def preprocess_single_image(filepath):
    pic = cv2.imread(filepath)
    pic = cv2.resize(pic, (120, 120))
    pic = pic.astype('float32')
    pic /= 255
    pic = pic.reshape(-1, 120, 120, 3)
    return pic

if __name__ == "__main__":
    server = WSGIServer(("", 5000), app)
    print('Server is up and running on port 5000...')
    print('Go to http://localhost:5000')
    server.serve_forever()
