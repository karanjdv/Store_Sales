import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Item_Identifiercy=eval(request.form['Item_Identifiercy'])
            Item_Weight=float(request.form['Item_Weight'])
            Item_Fat_Content=eval(request.form['Item_Fat_Content'])
            Item_Visibility=float(request.form['Item_Visibility'])
            Item_Type=eval(request.form['Item_Type'])
            Item_MRP=float(request.form['Item_MRP'])
            Outlet_Identifier=eval(request.form['Outlet_Identifier'])
            Outlet_Establishment_Year=eval(request.form['Outlet_Establishment_Year'])
            Outlet_Size=eval(request.form['Outlet_Size'])
            Outlet_Location_Type=eval(request.form['Outlet_Location_Type'])
            Outlet_Type=eval(request.form['Outlet_Type'])
            input_data=[Item_Identifiercy,Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Identifier,Outlet_Establishment_Year,Outlet_Size,Outlet_Location_Type,Outlet_Type]
            filename = 'model.pickle'
            loaded_model = pickle.load(open(filename, 'rb'))
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
            prediction=loaded_model.predict(input_data_reshaped)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('index.html',prediction=prediction)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__=="__main__":
    app.run(debug=True)

