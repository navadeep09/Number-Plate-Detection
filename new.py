import streamlit as st
import numpy as np
import cv2
import imutils
import pytesseract as pt
from io import BytesIO
from PIL import Image
import mysql.connector
import numpy as np
import matplotlib.pyplot as plt
from os.path import splitext,basename
from keras.models import model_from_json
from tensorflow.keras.utils import load_img ,img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder

json_file = open(r'C:\Users\navad\Downloads\IDP 3-1\MobileNets_character_recognition.json')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(r"C:\Users\navad\Downloads\IDP 3-1\License_character_recognition.h5")
labels = LabelEncoder()
labels.classes_ = np.load(r'C:\Users\navad\Downloads\IDP 3-1\license_character_classes.npy')

def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

def find_contours(dimensions, img) :
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    ii = cv2.imread('contour.jpg')
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX)
            char_copy = np.zeros((44,24))
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')
            char = cv2.subtract(255, char)
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) 
    plt.show()
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)
    return img_res
    
def segment_characters(image) :
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))
    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)
    char_list = find_contours(dimensions, img_binary_lp)
    return char_list

image = Image.open(r'logo.png') 
col1, col2 = st.columns( [0.8, 0.2])
with col1:               
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Upload your photo here...</p>', unsafe_allow_html=True)
with col2:
    st.image(image,  width=150)
st.sidebar.markdown('<p class="font">Number Plate Recognisation</p>', unsafe_allow_html=True)
st.sidebar.write(':car::car::car::car::car::car:')
with st.sidebar.expander("About the App"):
     st.write("""
        Use this simple app to recognize number plate ,extract the details of the vechile owner. 
         \n  
     """)
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
        st.image(image,width=300)  

    with col2:
        st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True)
        #image = imutils.resize(image, width=300 )
        image = np.array(image.convert('RGB'))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17) 
        edged = cv2.Canny(gray_image, 30, 200) 
        cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image1=image.copy()
        cv2.drawContours(image1,cnts,-1,(0,255,0),3)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30]
        screenCnt = None
        image2 = image.copy()
        cv2.drawContours(image2,cnts,-1,(0,255,0),3)
        i=7
        new_img = image.copy()
        for c in cnts:
                perimeter = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
                if len(approx) == 4: 
                        screenCnt = approx
                        x,y,w,h = cv2.boundingRect(c) 
                        new_img=image[y:y+h,x:x+w]
                        cv2.imwrite('./'+str(i)+'.png',new_img)
                        i+=1
                        break
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
        Cropped_loc = './7.png'
        st.image(new_img,width=300)
        imag = cv2.imread(r'7.png')
        char = segment_characters(imag)
        final_string = ''
        for i,character in enumerate(char):
            title = np.array2string(predict_from_model(character,model,labels))
            final_string+=title.strip("'[]")
    st.write("number plate of a vechile",final_string)
    re = []
    mydb = mysql.connector.connect(
    host="localhost",
    user="user",
    password="nani123",
    database="nprd"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM details")
    myresult = mycursor.fetchone()
    for x in myresult:
        re.append(x)
    st.write("name of the owner:        ",re[1])
    st.write("company of vechile:   ",re[2])
    st.write("Address of the owner:",re[3])
    
