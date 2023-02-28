from sklearn.neighbors import KernelDensity
import numpy as np

class OutlierDetector():
	def __init__(self, x, y, bandwidth):
		self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.hstack((x[:, np.newaxis], y[:, np.newaxis])))
		
		# determine cutoff
		predict_outlier = self.kde.score_samples(np.hstack((x[:, np.newaxis], y[:, np.newaxis])))
		ns, self.bins = np.histogram(predict_outlier)
		has_data = ns>0
		self.m, self.b = np.polyfit(self.bins[:-1][has_data], np.log(ns[has_data]), 1)
		self.outlier_cutoff = -self.b/self.m
		#print(self.bins, self.m, self.b, self.outlier_cutoff)

	def predict_outliers(self, x, y):
		predict_outlier = self.kde.score_samples(np.hstack((x[:, np.newaxis], y[:, np.newaxis])))
		return (predict_outlier<self.outlier_cutoff)

	def score_samples(self, x, y):
		return self.kde.score_samples(np.hstack((x[:, np.newaxis], y[:, np.newaxis])))
