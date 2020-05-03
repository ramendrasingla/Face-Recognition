import os
from flask import Flask, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from logging import DEBUG
from predictor import predict
import numpy as np
import webbrowser
app = Flask(__name__)
app.logger.setLevel(DEBUG)

dropzone = Dropzone(app)
# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'result'
# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(os.getcwd(),'static','uploads')
app.config['SECRET_KEY'] = 'supersecretkeygoeshere'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)
@app.route('/', methods=['GET','POST'])
def index():
	if "file_urls" not in session:
		session['file_urls'] = []
	if "result1" not in session:
		session['result1'] = []
	if "positive" not in session:
		session['positive'] = []
	full_result = session['result1']
	full_file = session['file_urls']
	full_positive = session['positive']
	if request.method == 'POST':
		file_obj = request.files
		for f in file_obj:
			file = request.files.get(f)
			filename = photos.save(file,name=file.filename)
			print(filename)
			result1,file_urls,positive = predict([filename])
			full_result = full_result+result1
			full_file = full_file+file_urls
			full_positive = full_positive+positive
		session['file_urls'] = full_file
		session['result1'] = full_result
		session['positive'] = full_positive
		print('COMPLETE')
		return "Uploading..."
	return render_template('index.html')
@app.route('/result')
def result():
	file_urls = session['file_urls']
	result1 = session['result1']
	positive = session['positive']
	session.pop('file_urls',None)
	session.pop('result1',None)
	session.pop('positive',None)
	print(positive)
	return render_template('result.html', file_urls=file_urls,result1=result1,len = len(file_urls),positive = positive)

if __name__ == '__main__':
	app.run(debug = True)
