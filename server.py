import os
from datetime import datetime
import hashlib
from flask import Flask, render_template, request, redirect, url_for

from sam_encoder_helper import encode_image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
EMBEDDING_FOLDER = 'static/upload_embeddings'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EMBEDDING_FOLDER'] = EMBEDDING_FOLDER

def generate_filename(file):
    return file.filename
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    extension = os.path.splitext(file.filename)[1]
    hash = hashlib.sha256()
    hash.update((timestamp + file.filename).encode('utf-8'))
    return hash.hexdigest() + extension

def get_embedding_filename(filename):
    return filename.rsplit(".", 1)[0] + ".bin"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = generate_filename(file)
    embedding_filename = get_embedding_filename(filename)

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    embedding_path = os.path.join(app.config['EMBEDDING_FOLDER'], embedding_filename)
    file.save(image_path)

    encode_image(image_path, embedding_path)

    return redirect(url_for('display_image', filename=filename))

@app.route('/display_image/<filename>')
def display_image(filename):
    embedding_filename = get_embedding_filename(filename)
    return render_template('interactive.html',
                           filename=filename,
                           embedding_filename=embedding_filename)


if __name__ == '__main__':
    app.run(debug=True)