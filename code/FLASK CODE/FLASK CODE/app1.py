from flask import Flask,render_template,request
import pickle
import numpy as np

import pandas as pd

model = pickle.load(open('pipe.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('credit.html')

@app.route('/Predict', methods= ['POST'])
def submit_score():

    # age = request.form.get('Enter your  Age')
    gender = request.form.get('Enter your the gender : Female:F/Male:M')
    car = request.form.get('Do you own car : Y/N')
    houses = request.form.get('Own houses :Y/N')
    occupation = request.form.get('Occupation of user')
    migrant = request.form.get('Migrant_worker :Y/N')
    credit_limit = request.form.get('Present Credit Limit')
    credit_percent = request.form.get('Credit limit used %')
    prev_default = request.form.get('No of prev defaults till now')
    six_month = request.form.get('Recent  default in last 6 months')
    children = request.form.get('Number of children')
    family_members = request.form.get('Total family members')
    yearly_debt = request.form.get('Yearly debt payment')
    credit_score = int(request.form.get('Credit_Score'))
    
    
    result= model.predict(pd.DataFrame(np.array([gender, car, houses, occupation, migrant, credit_limit, credit_percent, prev_default, six_month, children, family_members, yearly_debt, credit_score]).reshape(1, 13),columns=['gender', 'owns_car', 'owns_house', 'occupation_type',
       'migrant_worker', 'credit_limit', 'credit_limit_used(%)',
       'prev_defaults', 'default_in_last_6months',
       'no_of_children_', 'total_family_members_', 'yearly_debt_payments_',
       'credit_score_']))


    if result[0] == 1

         result ="YOU ARE ALLOWED TO GIVE CREDIT"


    else:
         result="CHANCE OF BEING  CREDIT DEFAULT IS VERY HIGH"

    return render_template('credit.html',result=result)




if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)