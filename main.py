from tkinter import *
from tkinter import messagebox
import tkinter.scrolledtext as tkst
import cv2
import PIL.Image, PIL.ImageTk
import time
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from timeit import default_timer as timer
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import tensorflow as tf


class App:
    emotions_text = ""

    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.iconbitmap(r'Asset_10.ico')
        # self.window.geometry("1075x556")
        self.window.configure(background="#716D6B")
        self.window.resizable(False, False)
        self.video_source = video_source

        # initialize menubar to main window
        menuBar = Menu(window)
        # initialize filemenu, exit command and add to menubar
        fileMenu = Menu(menuBar, tearoff=0)
        fileMenu.add_command(label="Exit", command=window.quit, accelerator="Ctrl+Q")
        menuBar.add_cascade(label="File", menu=fileMenu)

        # initialize viewmenu
        viewMenu = Menu(menuBar, tearoff=0)
        # initialize changeView submenu to viewmenu
        changeView = Menu(viewMenu, tearoff=0)
        changeView.add_command(label="Basic View", command=self.setNormal, accelerator="Ctrl+F")
        changeView.add_command(label="Facial Analysis", command=self.setDlib, accelerator="Ctrl+G")
        changeView.add_command(label="Emotion Detector", command=self.setEmotion, accelerator="Ctrl+H")
        # Add submenu to viewmenu
        viewMenu.add_cascade(label="Change View", menu=changeView)
        menuBar.add_cascade(label="View", menu=viewMenu)

        # initialize help menu
        helpMenu = Menu(menuBar, tearoff=0)
        helpMenu.add_command(label="About", command=self.aboutPanel, accelerator="Ctrl+I")
        menuBar.add_cascade(label="Help", menu=helpMenu)

        window.config(menu=menuBar)
        self.vid = MyVideoCapture(self.video_source)

        # Text box frame
        self.dialogFrame = Frame(window, borderwidth=1, relief="sunken")
        self.dialogFrame.grid(row=0, column=1, sticky=N + S + W, padx=(2, 2), pady=(8, 8))
        self.dataDisplay = tkst.ScrolledText(self.dialogFrame, wrap=NONE, borderwidth=1, width=50, height=30,
                                             state="normal", font='System 10 bold')
        self.dataDisplay.pack(fill=Y, expand=False)

        # south frame/button panel
        self.southFrame = Frame(window, borderwidth=1, relief="ridge", width=1075, background="#ABA6A4")
        self.southFrame.grid(row=1, column=0, columnspan=2, sticky=NSEW)

        self.btn_dlib = Button(self.southFrame, text="Data Analysis", width=20, command=self.setDlib)
        self.btn_dlib.configure(background="white", foreground="black")
        self.btn_dlib.pack(side=LEFT, padx=(100, 10), pady=(10, 10))

        self.btn_emotion = Button(self.southFrame, text="Emotion Detector", width=20, command=self.setEmotion)
        self.btn_emotion.configure(background="white", foreground="black")
        self.btn_emotion.pack(side=RIGHT, padx=(10, 100), pady=(10, 10))

        self.btn_normal = Button(self.southFrame, text="Normal Mode", width=20, command=self.setNormal)
        self.btn_normal.configure(background="#7EB4D4", foreground="black")
        self.btn_normal.pack(side=RIGHT, padx=(10, 200), pady=(10, 10))

        # VideoStream canvas
        self.canvas = Canvas(width=self.vid.width, height=self.vid.height)
        self.canvas.grid(row=0, column=0, sticky=NSEW, padx=(2, 2), pady=(8, 8))

        self.delay = 3
        self.update()

        self.window.mainloop()

    def setEmotionResults(self, emotion, emotion_probability):
        App.emotions_text = "Data output:\n" + "Currently predicting " + emotion + " with a " + "{0:.2f}".format(emotion_probability * 100) + "% accuracy"
    
    def setdlibResults(self, leftAspectRatio, rightAspectRatio):
        App.emotions_text = "Data output:\n" + "Left Eye Aspect Ratio : " + str(leftAspectRatio) + "\nRight Eye Aspect Ratio : " + str(rightAspectRatio)

    def getResults(self):
        return App.emotions_text

    def aboutPanel(self):
        messagebox.showinfo("About", "This is the about page for our project\n, talk about uses")

    def setNormal(self):
        self.btn_dlib.configure(background='white', foreground='black')
        self.btn_emotion.configure(background='white', foreground='black')
        self.btn_normal.configure(background='#7EB4D4', foreground='black')
        self.vid = MyVideoCapture(self.video_source)

    def setDlib(self):
        self.btn_normal.configure(background='white', foreground='black')
        self.btn_emotion.configure(background='white', foreground='black')
        self.btn_dlib.configure(background='#7EB4D4', foreground='black')
        self.vid = FacialCapture(self.video_source)

    def setEmotion(self):
        self.btn_normal.configure(background='white', foreground='black')
        self.btn_dlib.configure(background='white', foreground='black')
        self.btn_emotion.configure(background='#7EB4D4', foreground='black')
        self.vid = emotionCapture(self.video_source)

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
            self.dataDisplay.delete("1.0", END)
            self.dataDisplay.insert(INSERT, self.getResults())
        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


class FacialCapture(App):
    def __init__(self, video_source=0):
        self.video = cv2.VideoCapture(video_source)
        if not self.video.isOpened():
            raise ValueError("Unable to open video source", video_source)
        # Get video source width and height
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.video.isOpened():
            ret, image = self.video.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)  # detect faces in the grayscale frame
            for rect in rects:

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                #################################################################
                ############### Checking if eyes are closed/open ################
                #################################################################
                # grab the indexes of the facial landmarks for the left and
                # right eye, mouth respectively
                (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                ##### Get the left ear ####
                A = dist.euclidean(leftEye[1], leftEye[5])
                B = dist.euclidean(leftEye[2], leftEye[4])
                # compute the euclidean distance between the horizontal
                # eye landmark (x, y)-coordinates
                C = dist.euclidean(leftEye[0], leftEye[3])
                # compute the eye aspect ratio
                leftEAR = (A + B) / (2.0 * C)
                emotion_text = 'looking left'

                ##### Get the Right ear ####
                A = dist.euclidean(rightEye[1], rightEye[5])
                B = dist.euclidean(rightEye[2], rightEye[4])
                # compute the euclidean distance between the horizontal
                # eye landmark (x, y)-coordinates
                C = dist.euclidean(rightEye[0], rightEye[3])
                # compute the eye aspect ratio
                rightEAR = (A + B) / (2.0 * C)
                emotion_text = 'looking Right'

                number = (rightEAR + leftEAR) / 2

                App.setdlibResults(self, leftEAR, rightEAR)

                # loop over the (x, y)-coordinates for the facial landmarks
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


            if ret:
                return (ret, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.video.isOpened():
            self.video.release()


emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)
# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

# starting lists for calculating modes
emotion_window = []


class emotionCapture(App):
    emotion_classifier = load_model(emotion_model_path)
    graph = tf.get_default_graph()
    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    def __init__(self, video_source=0):
        self.video = cv2.VideoCapture(video_source)
        if not self.video.isOpened():
            raise ValueError("Unable to open video source", video_source)
        # Get video source width and height

    def get_frame(self):
        if self.video.isOpened():
            ret, image = self.video.read()

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                  minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            emotion_text = 'neutral'
            for face_coordinates in faces:
                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, self.emotion_target_size)
                except:
                    continue
                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                with self.graph.as_default():
                    emotion_prediction = self.emotion_classifier.predict(gray_face)
                    emotion_probability = np.max(emotion_prediction)
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_text = emotion_labels[emotion_label_arg]
                    emotion_window.append(emotion_text)
                    if len(emotion_window) > frame_window:
                        emotion_window.pop(0)
                try:
                    emotion_mode = mode(emotion_window)
                except:
                    continue

                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))

                App.setEmotionResults(self, emotion_text, emotion_probability)
                color = color.astype(int)
                color = color.tolist()

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, emotion_mode,
                          color, 0, -45, 1, 1)

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            data_emotion = np.concatenate((image, bgr_image), axis=1)

            if ret:
                return (ret, cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def __del__(self):
        if self.video.isOpened():
            self.video.release()


# Create a window and pass it to the Application object
App(Tk(), "GreenBit - tkinter, OpenCV, Dlib")