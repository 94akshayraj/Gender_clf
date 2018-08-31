import sklearn
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

classifiers =	[svm.SVC(),
				tree.DecisionTreeClassifier(),
				KNeighborsClassifier(3),
				SGDClassifier(loss="hinge", penalty="l2"),
				LinearDiscriminantAnalysis(),
				GaussianNB()]

names = ["SVM","DecisionTree","KNN","SGD","LDA","NaiveBayes"]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

for clf,name in zip(classifiers,names):

	clf = clf.fit(X,Y)
	prediction = clf.predict(X)
	print(name,"Accuracy:",sum(prediction==Y)/len(Y))
	# prdct = clff.predict([[190,70,43]])
