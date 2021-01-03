from flask import Flask
from flask_restful import Api, Resource, reqparse
import joblib
import numpy as np
import pandas as pd
import sys 
from flask import Flask, request, jsonify,render_template
import traceback
from flask_cors import CORS


APP = Flask(__name__)
api = Api(APP)


modelo = joblib.load('modelo_clustering.pkl')
modelo_columnas = joblib.load('modelo_columns_clustering.pkl')

@APP.route('/')
def home():
    return 'Hola Modelo de ML'

from flask import json

@APP.route('/predict', methods=['POST', 'GET'])
def predict():
    import pandas as pd 
    from sklearn.cluster import KMeans

    #data = request.json
    Edad = request.get_json('Edad')['Edad']
    Card_Mastercard_VERDADERO = request.get_json('Edad')['Card Mastercard_VERDADERO']
    Profesion_agrupacion_Dueña_de_casa = request.get_json('Edad')['Profesion_agrupacion_Dueña de casa']
    Profesion_agrupacion_Estudiante = request.get_json('Edad')['Profesion_agrupacion_Estudiante']
    Profesion_agrupacion_Trabajo_informado = request.get_json('Edad')['Profesion_agrupacion_Trabajo informado']
    Santiago = request.get_json('Edad')['Region Pais_Santiago']


    lista_datos = []
    lista_datos.append(Edad)
    lista_datos.append(Card_Mastercard_VERDADERO)
    lista_datos.append(Profesion_agrupacion_Dueña_de_casa)
    lista_datos.append(Profesion_agrupacion_Estudiante)
    lista_datos.append(Profesion_agrupacion_Trabajo_informado)
    lista_datos.append(Santiago)
    
    df = pd.DataFrame([lista_datos], columns = modelo_columnas)

    prediction = modelo.predict(df)

    output = round(prediction[0], 2)

    out_put_dic = {"predictions" : str(output)} 

    response = APP.response_class(
        response=json.dumps(out_put_dic),
        status=200,
        mimetype='application/json'
    )

    return response

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 12345 # If you don't provide any port then the port will be set to 12345
    modelo = joblib.load('modelo_clustering.pkl')
    modelo_columnas = joblib.load('modelo_columns_clustering.pkl') # Load "model.pkl"
    APP.run(port=port, debug=True)  



#if __name__ == '__main__':
#    APP.run(debug=True, port='1080')


