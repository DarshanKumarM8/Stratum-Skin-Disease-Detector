from flask import Flask


app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'app/static/uploads'
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Force template reloading
app.jinja_env.auto_reload = True  # Ensure Jinja reloads templates
from app import routes