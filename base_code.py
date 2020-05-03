from matplotlib import pyplot
from PIL import Image
from numpy import asarray
import numpy
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import pandas as pd
import os
import cvlib as cv
import cv2
static = os.path.join(os.getcwd(),'static')
base = os.path.join(static,'dataset')
emb_save = os.path.join(static,'emb')
extra = os.path.join(static,'extra')
face2 = os.path.join(static,'face')
cascPath = os.path.join(static,"haarcascade_frontalface_default.xml")
def extract_face(filename, required_size=(200, 200)):
    if filename[-3:]=='png':
        im = Image.open(os.path.join(base,filename)).convert('RGB')
        im.save(os.path.join(extra,(filename[:-3]+'jpg')))
        filename = filename[:-3]+'jpg'
        f1 = filename
        filename = os.path.join(extra,filename)
    elif filename[-4:]=='jpeg':
        im = Image.open(os.path.join(base,filename)).convert('RGB')
        im.save(os.path.join(extra,(filename[:-4]+'jpg')))
        filename = filename[:-4]+'jpg'
        f1 = filename
        filename = os.path.join(extra,filename)
    else:
        f1 = filename
        filename = os.path.join(base,filename)
    pixels1 = pyplot.imread(filename, 0)
    detector1 = cv
    detector2 = MTCNN()
    detector3 = cv2.CascadeClassifier(cascPath)
    results1,_ =  detector1.detect_face(pixels1)
    results2 = detector2.detect_faces(pixels1)
    results3 = detector3.detectMultiScale(pixels1,scaleFactor=1.1,minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE)
    if(len(results1)!=0):
        face1 = results1[0]
        x1,y1,x2,y2 = face1[0],face1[1],face1[2],face1[3]
        face = pixels1[y1:y2, x1:x2]
    elif(len(results2)!=0):
        x1, y1, width, height = results2[0]['box']
        x2, y2 = x1+width, y1+width
        face = pixels1[y1:y2, x1:x2]
    elif(len(results3)!=0):
        (x1,y1,w,h) = results3[0]
        x2,y2 = x1+w,y1+h
        face = pixels1[y1:y2, x1:x2]
    else:
        face = pixels1
    image = Image.fromarray(face)
    image.save(os.path.join(face2,f1))
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array
def get_embeddings(filenames):
    faces = [extract_face(f) for f in filenames]
    samples = asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50', include_top=False, input_shape=(200, 200, 3), pooling='avg')
    yhat = model.predict(samples)
    return yhat
def runn():
    filenames = os.listdir(base)
    embeddings = get_embeddings(filenames)
    img_file = pd.DataFrame(columns = ['Image_add','Emb_add'])
    im1 = []
    im2 = []
    for ix in range(len(filenames)):
        numpy.save(os.path.join(emb_save,filenames[ix]),embeddings[ix])
        im1.append(filenames[ix])
        im2.append(os.path.join(emb_save,filenames[ix])+'.npy')
    img_file['Image_add'] = numpy.array(im1)
    img_file['Emb_add'] = numpy.array(im2)
    img_file.to_csv('Address.csv',index = False)
    return

runn()
