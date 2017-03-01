#!/usr/bin/python
# -*- coding: latin-1 -*-

from mnist import MNIST
import sys
from random import randint
import datetime
import numpy as np

#pas sure
'''
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
'''
'''
from mnist import MNIST
mndata = MNIST('samples')
'''

#Importer La base de tensorflow et les datasets
#mnist, là ou sera stocker nos données de training, test, etc.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Interactive session sinn faudra d'abord faire le graphe avant de commencer la session
import tensorflow as tf
sess = tf.InteractiveSession()

#Des valeurs à donner(x, y_) lorsqu'on lancera le tensorflow
#none va correspondre au batch_size du input
#y_ va nous donner un vecteur qui indique à quel digit il appartient
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#on a des variables, de ce fait le model peut les utiliser et les changer au fur et à mesure initialisé à Zero :)
#w le weight, le poids en couleur
#b represente le bias
W = tf.Variable(tf.zeros([784,10])) #Parcequ'on a de base 784 et on veut du 10
b = tf.Variable(tf.zeros([10])) #parcequ'on a 10 classes

#Change les valeurs initiales pour leur donner leurs vraies valeurs
sess.run(tf.global_variables_initializer())

#Modelde regression: mul de nos entrées avec leur poids et addition avec le bias
y = tf.matmul(x,W) + b

#Calcule le loss: la différence entre la prédiction et le réel
#Prends la Moyenne du resultat du softmax
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#Partie training: utilisation du Gradient pour optimizer et réduire le loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #Pas de 0.5

#Repeter le training
for _ in range(1000):
  batch = mnist.train.next_batch(100) #100 images de sets à chaque iteration/epoch
  train_step.run(feed_dict={x: batch[0], y_: batch[1]}) #feed_dict pr ecrire dans x et y_ les elements du mnist set

#Test: entre ce qu'il a trouvé et ce qui est réel => retourne un Booléen
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#Transformation en float des booleens et fais le pourcentage on va dire
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Evaluation Géneral de notre model avec les images d'mnist et les lables d'mnist
print("Apprentissage FINI!")
print("Resultat: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#PART1

#test du model sur une image MNIST quelconque du set original
num = randint(0, mnist.test.images.shape[0]) #Pour trouver un nombre random
img = mnist.test.images[num] #Pour prendre une image Random

classification = sess.run(tf.argmax(y, 1), feed_dict={x: [img]})

print ("Prediction: ", classification[0])
print ("Realite: ", np.argmax(mnist.test.labels[num]))

mndata = MNIST('MNIST_data')

images, labels = mndata.load_testing()

print(mndata.display(images[num]))

#PART2
'''
#2: Using our model to classify MNIST digit from a custom image:

# create an an array where we can store 1 picture
images = np.zeros((1,784))
# and the correct values
correct_vals = np.zeros((1,10))

# read the image
gray = cv2.imread("my_digit.png", 0 ) #0=cv2.CV_LOAD_IMAGE_GRAYSCALE #must be .png!

# rescale it
gray = cv2.resize(255-gray, (28, 28))

# save the processed images
cv2.imwrite("my_grayscale_digit.png", gray)
"""
all images in the training set have an range from 0-1
and not from 0-255 so we divide our flatten images
(a one dimensional vector with our 784 pixels)
to use the same 0-1 based range
"""
flatten = gray.flatten() / 255.0
"""
we need to store the flatten image and generate
the correct_vals array
correct_val for a digit (9) would be
[0,0,0,0,0,0,0,0,0,1]
"""
images[0] = flatten


my_classification = sess.run(tf.argmax(y, 1), feed_dict={x: [images[0]]})

"""
we want to run the prediction and the accuracy function
using our generated arrays (images and correct_vals)
"""
print 'Neural Network predicted', my_classification[0], "for your digit"
'''
