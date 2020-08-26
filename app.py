from flask import Flask,render_template,url_for,request,session,redirect,g
from flask_material import Material

# import EDA Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import ML Packages

# joblib or pickle are for loading the model (already built)
from sklearn.externals import joblib 


app=Flask(__name__)
Material(app)

@app.route('/')

def index():
	return render_template("index.html")


@app.route('/preview')

def preview():
	df=pd.read_csv("C:\\Users\\BENTEKER\\Documents\\Ben_ML_Deployement_Example\\Iris_deployement\\data\\iris.csv")
	return render_template("preview.html",df_view=df)

import seaborn as sns

@app.route('/graph')

def graph():
	df=pd.read_csv("C:\\Users\\BENTEKER\\Documents\\Ben_ML_Deployement_Example\\Iris_deployement\\data\\iris.csv")
	#grp=sns.pairplot(data=df,hue='species')
	grp=df.describe().T
	return render_template("graph.html",grp_view=grp)

@app.route('/vis')

def vis():
	df=pd.read_csv("C:\\Users\\BENTEKER\\Documents\\Ben_ML_Deployement_Example\\Iris_deployement\\data\\iris.csv")
	graph = plt.scatter(list(df[['sepal_length']]), list(df[['sepal_width']]))
	#fig = graph.get_figure()
	filepath = "/static/graph.PNG"
	plt.figure(figsize=(15,15))
	plt.savefig(filepath)
	return render_template("vis.html")

@app.route('/analyze',methods=['POST'])

def analyze():
	if request.method=='POST':
		petal_length=request.form['petal_length']
		sepal_length=request.form['sepal_length']
		petal_width=request.form['petal_width']
		sepal_width=request.form['sepal_width']
		model_choice=request.form['model_choice']

		# Name of variable from dataset
		sample_data=[sepal_length,sepal_width,petal_length,petal_width] 

		# unicode to float
		clean_data=[float(i) for i in sample_data]

		# Reshaping:
		ex1=np.array(clean_data).reshape(1,-1)

		# ML conditional: 
		if model_choice=='logitmodel':
			logit_model=joblib.load('C:\\Users\\BENTEKER\\Documents\\Ben_ML_Deployement_Example\\Iris_deployement\\data\\logit_model_iris.pkl')
			result_prediction=logit_model.predict(ex1)
		elif model_choice=='knnmodel':
			knn_model=joblib.load('C:\\Users\\BENTEKER\\Documents\\Ben_ML_Deployement_Example\\Iris_deployement\\data\\knn_model_iris.pkl')
			result_prediction=knn_model.predict(ex1)
		elif model_choice=='svmmodel':
			svm_model=joblib.load('C:\\Users\\BENTEKER\\Documents\\Ben_ML_Deployement_Example\\data\\Iris_deployement\\svm_model_iris.pkl')
			result_prediction=svm_model.predict(ex1)
		else:
			logit_model=joblib.load('C:\\Users\\BENTEKER\\Documents\\Ben_ML_Deployement_Example\\data\\Iris_deployement\\logit_model_iris.pkl')
			result_prediction=logit_model.predict(ex1)


	return render_template("index.html",sepal_width=sepal_width, sepal_length=sepal_length,
		petal_length=petal_length,petal_width=petal_width,
		clean_data=clean_data,result_prediction=result_prediction,model_selected=model_choice)

if __name__ == '__main__':
	app.run(debug=True)