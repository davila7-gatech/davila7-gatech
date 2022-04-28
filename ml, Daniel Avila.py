import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import time
from skimage import data, color
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import cv2
import matplotlib.image as img
import imageio
from PIL import Image
import smtplib
server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()
server.login("ldmldavila@gmail.com", "D202214011225")# logon to gmail account for texting
import csv
import datetime
import os
from IPython.display import clear_output

def convertImage(img):
  blur = cv2.GaussianBlur(img, (3,3), 0)
  sat = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)[:,:,1]# convert image to hsv and get saturation channel
  thresh = cv2.threshold(sat, 50, 255, cv2.THRESH_BINARY)[1]# threshold saturation channel

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
  morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
  mask = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)# apply morphology close and open

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]# do OTSU threshold to get circuit image

  otsu_result = otsu.copy()
  otsu_result[mask==0] = 0# write black to otsu image
  return otsu_result

def centroidHistogram(clt):
  Labels = np.arange(0, len(np.unique(clt.labels_))+1)# grab the number of clusters and generate a histogram
  (hist, _) = np.histogram(clt.labels_, bins = Labels)
  hist = hist.astype("float")# normalize the histogram so that its sum is one
  return hist/hist.sum()

cv2.namedWindow("Location 1")
cv2.namedWindow("Location 2")
vc1 = cv2.VideoCapture(0)
vc2 = cv2.VideoCapture(2)

if vc1.isOpened() and vc2.isOpened():# try to get the first frame
  rval1, frame1 = vc1.read()
  rval2, frame2 = vc2.read()
  time.sleep(1)
  rval1, frame1 = vc1.read()
  rval2, frame2 = vc2.read()
  frame1 = convertImage(frame1)
  frame2 = convertImage(frame2)
  orig1 = frame1
  clt = KMeans(n_clusters=2)
  clt.fit(orig1)
  histOrig1 = centroidHistogram(clt)
  print('original1 histogrm: ', sorted(histOrig1))# histogram for original joint 1
  orig2 = frame2
  clt = KMeans(n_clusters=2)
  clt.fit(orig2)
  histOrig2 = centroidHistogram(clt)
  print('original2 histogrm: ', sorted(histOrig2))# histogram for original joint 2
else:
  rval1 = False
  rval2 = False

while rval1 and rval2:
  time.sleep(.2)
  rval1, frame1 = vc1.read()# capture frame from Location 1
  rval2, frame2 = vc2.read()# capture frame from Location 2
  frame1 = convertImage(frame1)
  frame2 = convertImage(frame2)
  cv2.imshow('Location 1', frame1)
  cv2.imshow('Location 2', frame2)# call the ML algorithm to compare the current image to the original image
  current1 = frame1
  clt = KMeans(n_clusters=2)
  clt.fit(current1)
  histCurrent1 = centroidHistogram(clt)# create histogram for the current joint 1
  current2 = frame2
  clt = KMeans(n_clusters=2)
  clt.fit(current2)
  histCurrent2 = centroidHistogram(clt)# create histogram for the current joint 2
  print('Location 1 Current centroids: ', sorted(histCurrent1))# test Joint 1 first
  anomalyCount1 = 0
  for i in range(1,4):
    if abs(np.max(histOrig1)-np.max(histCurrent1))/np.max(histOrig1) > .139:
      # determine whether joint 1 has leakage (modify this proportion to adjust sensitivity of camera 1)
      anomalyCount1 += 1
      time.sleep(.01)
      rval1, frame1 = vc1.read()
      current1 = convertImage(frame1)# process this frame
      clt = KMeans(n_clusters=2)
      clt.fit(current1)
      histCurrent1 = centroidHistogram(clt)# create histogram for the current joint 1
      print('Location 1 # of anomalies: ', anomalyCount1)
    if anomalyCount1 > 1:
      print('Location 1 leakage: ', np.max(histCurrent1), '; Original value: ', np.max(histOrig1))
      currentTime = datetime.datetime.now()
      timeStamp = currentTime.timestamp()
      server.sendmail('the system', '9546965906@tmomail.net', 'Plant Location 1 has leakage.')
      with open("log.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Location 1 has leakage.', timeStamp])
      img = cv2.imread('leakwarning.jpg', cv2.IMREAD_ANYCOLOR)
      cv2.imshow('Location 1', img)
      input('Press enter to continue.')
      break
    else:
      print(' ')

  print('Location 2 Current centroids: ', sorted(histCurrent2))# test Joint 2 next
  anomalyCount2 = 0
  for i in range(1,4):
    if abs(np.max(histOrig2)-np.max(histCurrent2))/np.max(histOrig2) > .139:
      # determine whether joint 2 has leakage (modify this proportion to adjust sensitivity of camera 2)
      anomalyCount2 += 1
      time.sleep(.01)
      rval2, frame2 = vc2.read()
      current2 = convertImage(frame2)# process this frame
      clt = KMeans(n_clusters=2)
      clt.fit(current2)
      histCurrent2 = centroidHistogram(clt)# create histogram for the current joint 2
      print('Location 2 # of anomalies: ', anomalyCount2)
    if anomalyCount2 > 1:
      print('Location 2 leakage: ', np.max(histCurrent2), '; Original value: ', np.max(histOrig2))
      currentTime = datetime.datetime.now()
      timeStamp = currentTime.timestamp()
      server.sendmail('the system', '9546965906@tmomail.net', 'Plant Location 2 has leakage.')
      with open("log.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Location 2 has leakage.', timeStamp])
      img = cv2.imread('leakwarning.jpg', cv2.IMREAD_ANYCOLOR) 
      cv2.imshow('Location 2', img)
      input('Press enter to continue.')
      break
    else:
      print(' ')

    key = cv2.waitKey(20)
    if key == 27:# exit on ESC
      break

input('Press Enter to acknowledge.')
vc1.release()
cv2.destroyWindow("Location 1")
vc2.release()
cv2.destroyWindow("Location 2")
