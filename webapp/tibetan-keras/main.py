import numpy as np
from flask import Flask,jsonify,render_template,request
from keras.models import load_model

lr_model = load_model('mnist/data/regression.hdf5')
lr_model.predict(np.zeros((1, 784)))
cnn_model = load_model('mnist/data/convolutional.hdf5')
cnn_model.predict(np.zeros((1, 28, 28, 1)))



def regression(input):
    pred = lr_model.predict(input.reshape(1, 784))
    return pred.flatten().tolist()


def convolutional(input):
    pred = cnn_model.predict(input.reshape(1, 28, 28, 1))
    return pred.flatten().tolist()


app=Flask(__name__)
@app.route('/api/mnist',methods=['post'])
def mnist():
    input=((255-np.array(request.json,dtype=np.uint8))/255.0).reshape(1,784)
    output1=regression(input)
    output2=convolutional(input)
    return jsonify(results=[output1,output2])

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug=True
    app.run()