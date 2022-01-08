from flask import Flask, render_template, request
import joblib
app = Flask(__name__)
model=joblib.load('hiring_model.pkl')
@app.route('/')
def new():
    return 'New Page'
@app.route('/welcome')
def welcome():
    return render_template('base.html')
@app.route('/predict', methods=['GET','POST'])    #instead of predict, whatever name we want we can give
def html_new():
    exp=request.form.get('experience')
    t_score=request.form.get('score')
    i_score=request.form.get('i_score')
    print(exp)
    print(t_score)
    print(i_score)
    prediction=model.predict([[int(exp),int(t_score), int(i_score)]])
    output=round(prediction[0],2)
    return render_template('base.html', prediction_text=f" Employee salary ill be {output}")
app.run(debug=True)
