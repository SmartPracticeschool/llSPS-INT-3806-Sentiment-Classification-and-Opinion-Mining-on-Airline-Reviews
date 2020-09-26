from flask import render_template, Flask, request,url_for
from tensorflow.keras.models import load_model
import pickle 
with open(r'CountVectorizer','rb') as file:
    cv=pickle.load(file)
cla = load_model('airline_predictions.h5')
cla.compile(optimizer='adam',loss='binary_crossentropy')
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('base.html')
@app.route('/prediction',methods=['GET'])
def prediction():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    if request.method == 'POST':
        topic = request.form['Review']
        topic=cv.transform([topic])
        print("\n"+str(topic.shape)+"\n")
        y_pred = cla.predict_classes(topic)
        print("pred is "+str(y_pred))
        if(y_pred[0]==2):
            topic = "Positive Tweet"
        elif(y_pred[0]==0):
            topic = "Negative Tweet"
        else:
            topic = "Neutral Tweet"
        return render_template('index.html',prediction_text='The sentiment value is : {}'.format(topic))
        



if __name__ == '__main__':
    app.run(host = 'localhost', debug = True , threaded = False)
    
