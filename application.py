from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline # type: ignore


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            LIMIT_BAL=float(request.form.get('LIMIT_BAL')),
            SEX = int(request.form.get('SEX')),
            EDUCATION = int(request.form.get('EDUCATION')),
            MARRIAGE = int(request.form.get('MARRIAGE')),
            AGE = int(request.form.get('AGE')),
            PAY_1 = int(request.form.get('PAY_1')),
            PAY_2 = int(request.form.get('PAY_2')),
            PAY_3 = int(request.form.get('PAY_3')),
            PAY_4 = int(request.form.get('PAY_4')),
            PAY_5 = int(request.form.get('PAY_5')),
            PAY_6 = int(request.form.get('PAY_6')),
            BILL_AMT1 = float(request.form.get('BILL_AMT1')),
            BILL_AMT2 = float(request.form.get('BILL_AMT2')),
            BILL_AMT3 = float(request.form.get('BILL_AMT3')),
            BILL_AMT4 = float(request.form.get('BILL_AMT4')),
            BILL_AMT5 = float(request.form.get('BILL_AMT5')),
            BILL_AMT6 = float(request.form.get('BILL_AMT6')),
            PAY_AMT1 = float(request.form.get('PAY_AMT1')),
            PAY_AMT2 = float(request.form.get('PAY_AMT2')),
            PAY_AMT3 = float(request.form.get('PAY_AMT3')),
            PAY_AMT4 = float(request.form.get('PAY_AMT4')),
            PAY_AMT5 = float(request.form.get('PAY_AMT5')),
            PAY_AMT6 = float(request.form.get('PAY_AMT6'))
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        msg=''

        if results==0:
            msg='Will not Default'
        elif results==1:
            msg='Will Default'

        return render_template('form.html',final_result=msg)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)