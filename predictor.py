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
import cv2
import cvlib as cv
base = os.path.join(os.getcwd(),'static','uploads')
face2 = os.path.join(os.getcwd(),'static','face')
def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped
def extract_face(filename, required_size=(200, 200)):
    im = Image.open(os.path.join(base,filename)).convert('RGB')
    if filename[-3:]=='png':
        im.save(os.path.join(base,(filename[:-3]+'jpg')))
        filename = filename[:-3]+'jpg'
    elif filename[-4:]=='jpeg':
        im.save(os.path.join(base,(filename[:-4]+'jpg')))
        filename = filename[:-4]+'jpg'
    else:
        im.save(os.path.join(base,filename))
    pixels1 = pyplot.imread(os.path.join(base,filename))
    detector1 = cv
    detector2 = MTCNN()
    results1,_ =  detector1.detect_face(pixels1)
    results2 = detector2.detect_faces(pixels1)
    if(len(results1)!=0):
        face1 = results1[0]
        x1,y1,x2,y2 = face1[0],face1[1],face1[2],face1[3]
        face = pixels1[y1:y2, x1:x2]
    elif(len(results2)!=0):
        x1, y1, width, height = results2[0]['box']
        x2, y2 = x1+width, y1+width
        face = pixels1[y1:y2, x1:x2]
    else:
        face = pixels1
    image = Image.fromarray(face)
    image.save(os.path.join(face2,filename))
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
def predict(images):
    dataset = pd.read_csv('Address.csv',encoding = 'latin-1')
    dataset_emb = []
    for ix in range(dataset.shape[0]):
        a = numpy.load(dataset['Emb_add'][ix])
        dataset_emb.append(a)
    dataset_emb = numpy.array(dataset_emb)
    curr_images = get_embeddings(images)
    result = pd.DataFrame(columns=['Upload_img','Result_img','Dist'])
    for ix in range(len(images)):
        a = []
        for iy in range(dataset.shape[0]):
            a.append(cosine(curr_images[ix],dataset_emb[iy]))
        a = numpy.array([numpy.array(a),dataset['Image_add']])
        temp = pd.DataFrame(columns=['Upload_img','Result_img','Dist'])
        temp['Dist'] = a[0]
        temp['Result_img'] = a[1]
        temp['Upload_img'] = images[ix]
        result = pd.concat([result,temp],axis=0)
    result = result.sort_values(by = ['Dist'], ascending = True).groupby('Upload_img').head(3).reset_index().sort_values(by = ['Upload_img'], ascending = True)
    result['Positive'] = result['Dist'].apply(lambda x:1 if(x<0.52) else 0)
    df_positive = result[['Upload_img','Positive']].groupby('Upload_img').sum().reset_index().sort_values(by = ['Upload_img'], ascending = True)
    return list(result['Result_img']),list(result['Upload_img'].drop_duplicates()),list(df_positive['Positive'])
