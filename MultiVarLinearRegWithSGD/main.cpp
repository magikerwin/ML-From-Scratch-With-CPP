/*
Multivariate linear regression with SGD algorithm:

1. estimate coefficients with SGD
    1.1 init weights
    1.2 get loss
    1.3 update weights and bias
        -- bias = bias - lr * (y_hat - y)
        -- weights = weight - lr * (y_hat - y) * x
2. make predicitons
*/

#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

double predict(
    const vector<double> &xs, 
    const vector<double> &coefficents) {
    
    double prediction = coefficents[0];
    for (int i=0; i<xs.size(); i++) {
        prediction += coefficents[i+1] * xs[i];
    }
    return prediction;
}

vector<double> estimateCoefficientsWithSGD(
    const vector<vector<double>> trainSet, 
    const double lr, const int numOfEpochs) {
    
    // init weights
    int numOfCoefficents = (trainSet[0].size() - 1) + 1;
    vector<double> coefficents(numOfCoefficents, 0);
    
    for (int e=0; e<numOfEpochs; e++) {
        double sumOfError = 0;
        for (auto pair : trainSet) {
            vector<double> x = {pair[0]};
            double prediction = predict(x, coefficents);
            double error = pow(pair[1] - prediction, 2);
            sumOfError += error;
            
            // update
            //// bias = bias - lr * (y_hat - y)
            coefficents[0] = coefficents[0] - lr * (prediction - pair[1]);
            //// weights = weight - lr * (y_hat - y) * x
            for (int i=1; i<numOfCoefficents; i++) {
                coefficents[i] = coefficents[i] - lr * (prediction - pair[1]) * pair[0];
            }
        }
        
        cout << "epoch: " << e
            << ", error: " << sumOfError << endl;
    }
    
    return coefficents;
}

int main() {
    
    vector<vector<double>> trainSet = {
        {1, 1},
        {2, 3},
        {4, 3},
        {3, 2},
        {5, 5}
    };
    
    // vector<double> coefficents = {0.4, 0.8};
    double lr = 0.001;
    int numOfEpochs = 50;
    vector<double> coefficents = estimateCoefficientsWithSGD(trainSet, lr, numOfEpochs);
    cout << "coefficents = {" 
        << coefficents[0] << ", "
        << coefficents[1] << "}"<< endl;
    
    for (auto pair : trainSet) {
        cout << predict({pair[0]}, coefficents) << endl;
    }
}