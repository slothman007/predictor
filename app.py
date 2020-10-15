from flask import Flask,render_template,request
import pickle
import pandas as pd
app=Flask(__name__)
dt=pickle.load(open('model.pkl','rb'))
train = pd.read_csv("Training.csv")
X = train.drop(['prognosis'],axis=1)
S=list()
for i in range(len(X.columns)):
	S.append(X.columns[i])
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
	if request.method == 'POST':
		symps = request.form.getlist('symps')
		a=[0]*len(S)
		for p in symps:
			p=p.replace(' ','_')
			n=S.index(p)
			a[n]=1
		y_diagnosis = dt.predict([a])
		y_pred_2 = dt.predict_proba([a])
		print(('Name of the infection = %s , confidence score of : = %s') %(y_diagnosis[0],y_pred_2.max()* 100),'%' )
		return render_template("index.html",result = y_diagnosis[0],per=y_pred_2.max()* 100)

dt=pickle.load(open('model.pkl','rb'))
train = pd.read_csv("Training.csv")
X = train.drop(['prognosis'],axis=1)
X.head()
if(__name__=='__main__'):
	app.run()