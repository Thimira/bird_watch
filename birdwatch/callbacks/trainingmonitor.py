# import the necessary packages
from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
	def __init__(self, figPath, jsonPath=None, startAt=0):
		# store the output path for the figure, the path to the JSON
		# serialized file, and the starting epoch
		super(TrainingMonitor, self).__init__()
		self.figPath = figPath
		self.jsonPath = jsonPath
		self.startAt = startAt

	def on_train_begin(self, logs={}):
		# initialize the history dictionary
		self.H = {}

		# if the JSON history path exists, load the training history
		if self.jsonPath is not None:
			if os.path.exists(self.jsonPath):
				self.H = json.loads(open(self.jsonPath).read())

				# check to see if a starting epoch was supplied
				if self.startAt > 0:
					# loop over the entries in the history log and
					# trim any entries that are past the starting
					# epoch
					for k in self.H.keys():
						self.H[k] = self.H[k][:self.startAt]

	def on_epoch_end(self, epoch, logs={}):
		# loop over the logs and update the loss, accuracy, etc.
		# for the entire training process
		for (k, v) in logs.items():
			l = self.H.get(k, [])
			l.append(v)
			self.H[k] = l

		# check to see if the training history should be serialized
		# to file
		if self.jsonPath is not None:
			# print(self.H)
			f = open(self.jsonPath, "w")
			f.write(json.dumps(self.H))
			f.close()

		# ensure at least two epochs have passed before plotting
		# (epoch starts at zero)
		if len(self.H["loss"]) > 1:
			# plot the training loss and accuracy

			N = np.arange(0, len(self.H["loss"]))
			
			plt.rcParams["figure.figsize"] = (12, 9)
			plt.style.use("ggplot")
			plt.figure(1)

			# subplot for accuracy
			plt.subplot(211)
			plt.plot(N, self.H["acc"], label="Training Accuracy")
			if 'val_acc' in self.H:
				plt.plot(N, self.H["val_acc"], label="Validation Accuracy")
			
			plt.title("Model Accuracy [Epoch {}]".format(len(self.H["loss"])))
			plt.ylabel('Accuracy')
			plt.xlabel('Epoch #')
			plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')

			# subplot for loss
			plt.subplot(212)
			plt.plot(N, self.H["loss"], label="Training Loss")
			if 'val_loss' in self.H:
				plt.plot(N, self.H["val_loss"], label="Validation Loss")

			plt.title("Model Loss [Epoch {}]".format(len(self.H["loss"])))
			plt.ylabel('Loss')
			plt.xlabel('Epoch #')
			plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')

			plt.tight_layout()

			# save the figure
			plt.savefig(self.figPath)
			plt.close()
