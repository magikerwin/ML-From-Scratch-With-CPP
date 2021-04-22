/*
logistic regression with SGD algorithm:

loss: l2-norm
activation function: sigmoid

1. estimate coefficients with SGD
    1.1 init weights
    1.2 get loss
    1.3 update weights and bias
        b = b - lr * (y_hat - y) * dSigmoid(z)
          = b - lr * (y_hat - y) * (y_hat * (1-y_hat))
        w = w - lr * (y_hat - y) * dSigmoid(z) * x
          = w - lr * (y_hat - y) * (y_hat * (1-y_hat)) * x
2. make predicitons
*/

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double predict(
    const vector<double> &x, 
    const vector<double> &coefficents) {
    
    int numOfInputDims = x.size() - 1;
    
    double z = coefficents[0];
    for (int i=0; i<numOfInputDims; i++) {
        z += coefficents[i+1] * x[i];
    }
    return sigmoid(z);
}

vector<double> estimateCoefficientsWithSGD(
    const vector<vector<double>> trainSet, 
    const double lr, const int numOfEpochs) {
    
    // init weights
    int numOfCoefficents = (trainSet[0].size() - 1) + 1;
    vector<double> coefficents(numOfCoefficents, 0);
    
    for (int e=0; e<numOfEpochs; e++) {
        double sumOfError = 0;
        for (auto data : trainSet) {
            double label = data[data.size() - 1];
            double prediction = predict(data, coefficents);
            double error = pow(label - prediction, 2);
            sumOfError += error;
            
            // update
            //// b = b - lr * (y_hat - y) * (y_hat * (1-y_hat))
            coefficents[0] = coefficents[0] - lr * (prediction - label) * prediction * (1 - prediction);
            //// w - lr * (y_hat - y) * (y_hat * (1-y_hat)) * x
            for (int i=1; i<numOfCoefficents; i++) {
                coefficents[i] = coefficents[i] - lr * (prediction - label) * prediction * (1 - prediction) * data[i-1];
            }
        }
        
        cout << "epoch: " << e
            << ", error: " << sumOfError << endl;
    }
    
    return coefficents;
}

int main() {
    
    vector<vector<double>> trainSet = {
        {2.7810836,2.550537003,0},
        {1.465489372,2.362125076,0},
        {3.396561688,4.400293529,0},
        {1.38807019,1.850220317,0},
        {3.06407232,3.005305973,0},
        {7.627531214,2.759262235,1},
        {5.332441248,2.088626775,1},
        {6.922596716,1.77106367,1},
        {8.675418651,-0.242068655,1},
        {7.673756466,3.508563011,1}
    };
    
    //vector<double> coefficents = {-0.406605464, 0.852573316, -1.104746259};
    double lr = 0.3;
    int numOfEpochs = 100;
    vector<double> coefficents = estimateCoefficientsWithSGD(trainSet, lr, numOfEpochs);
    
    // 
    cout << "coeff:" << endl;
    for (auto val : coefficents) cout << val << endl;
}