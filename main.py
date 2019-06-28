import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

#Labels as numbers
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def showPrediction(predictions, images, labels, number):
	plt.figure(figsize=(20,10))
	im = 1
	for i in range(number**2):
		
		plt.subplot(number, number*2, im)
		im += 1

		predictions_array, true_label, img = predictions[i], labels[i], images[i]
		plt.xticks([])
		plt.yticks([])
		plt.imshow(img, cmap=plt.cm.binary)
		
		predicted_label = np.argmax(predictions_array)

		if predicted_label == true_label:
			color ='blue'
		else:
			color ='red'

		plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
		  	100*np.max(predictions_array),
		  	class_names[true_label]),
		  	color=color)

	
		plt.subplot(number, number*2, im)
		im +=1
		
		thisplot = plt.bar(range(10), predictions_array, color="#777777")
		plt.xticks([])
		plt.yticks([])
		
		predicted_label = np.argmax(predictions_array)
		
		thisplot[predicted_label].set_color('red')
		thisplot[true_label].set_color('blue')

	plt.show()

def showImages(images, labels, size):
	plt.figure(figsize=(10,10))

	for i in range(size**2):
		plt.subplot(size, size, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(images[i], cmap='gray')
		plt.xlabel(class_names[labels[i]])

	plt.show()

#Retrieve Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(trainImages, trainLabels), (testImages, testLabels) = fashion_mnist.load_data()

##### Run Your Code #####

#STEP(1): Visualization/Validation

# TODO: Print the datatype of the dataset that you will use.
# Hint: Use type(Object) to get the type of the data that you'll work on.




# Solution: print('Datatype : ', type(trainImages))

#Ensure that your label is specified correctly.

# TODO: Print the maximum and the minmum of the train data labels to
# ensure that the whole labels in range between 0 --> 9
# Hint: Numpy array has two functions max()/min() to catch the highest
# and lowest value in the existed array.





# Solution: print('Max label: {}, Min label: {}'.format(trainLabels.max(), trainLabels.min()))

# Remove the comment symbol '#' from the bellow line
#showImages(trainImages, trainLabels,5)

###### Run your code ######

# TODO: Display the shape of the training and test dataset
# Hint: Numpy arrays have shape() function to display the structure of the list/s





#Solution: print('Data shape: ', trainImages.shape)
#Solution: print('Data shape: ', testImages.shape)


#STEP(2): Preprocessing
# Use the copies of train/test dataset to convert all gray images to binary images
# TODO: Convert gray images to binary image in range 0 ~ 1.
# Hint: Divide each dataset by 255
trainImages_cp = np.copy(trainImages)
testImages_cp  = np.copy(testImages)


trainImages_cp 	= 
testImages_cp 	= 




''' Solution:
trainImages_cp = trainImages_cp/ 255.0
testImages_cp = testImages_cp/ 255.0
'''

#STEP(3): Model Structure & Settings

#TODO: Determine the number of nodes for the input/hidden/output layers
#Hint: You can use these values as input/output nodes in order: 10 - 160
output_layer = 
hidden_layer = 


#TODO: Build your model's structure using Tensor.Keras API
#The more you increase the number of hidden layers, the more model train's time will be increased







'''
Solution:
model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28,28)),
		keras.layers.Dense(hidden_layer, activation=tf.nn.relu),
		keras.layers.Dense(output_layer, activation=tf.nn.softmax)
	])
'''

#TODO: Define the settings(Lossfunction, optimizer, metrics) of your model using compile method
#Hint: Use these settings :
#		Optimizer: adam | Loss Function: Cross-Entropy Loss | Metrics to evaluated your model: Accuracy








'''
Solution:
model.compile(
	optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
	)
'''

#STEP(4): Train process
# TODO: Assign fit number of epochs to train your model.
# Hint: Epochs are the number of repeatation for the training process in your model.
# 5 is acceptable.

epochs = 

# TODO: Start the training by calling fit function from your model.






'''
Solution:
model.fit(trainImages, trainLabels, epochs=epochs)
'''

#STEP(5): Evaluation.
# TODO: Evaluate your model using testing dataset
# Hint: Call evaluate function from your model object.






'''
Solution:
test_loss, test_acc = model.evaluate(testImages, testLabels)
'''

#TODO: Print both of model's accuracy and cost to visualize the 
# efficiency of your model






# Solution: print('Model Accuracy: ', test_acc, ' Model Cost: ', test_loss)

#STEP(6): Prediction for new inputs.
# TODO: Pass the new input dataset to predict function.





'''
Solution:
predictions = model.predict(testImages)
'''
#Remove the comment symbol from the below line.

#showPrediction(predictions, testImages, testLabels, 4)




