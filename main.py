from flask import Flask , request , jsonify
from model_files.ml_model import predict_mpg
import pickle


app = Flask('Mile-prediction')

# @app.route('/' , methods = ['GET'])
# def ping():
#     return 'Pinging Model App'

@app.route('/' , methods=['POST'])
def predict():
    vehicle_config = request.get_json()

    with open('./model_files/model.bin' , 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    
    predictiton = predict_mpg(vehicle_config , model)

    response = {
        'mpg_predictions': list(predictiton)

    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug = False ,host = '127.0.0.1' , port = 9696 )

