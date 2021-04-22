/*
perceptron algorithm:

loss: L2 norm
activation: 0/1

1. get loss
2. update weights
3. make predictions

*/

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;


double predict(vector<double> data, const vector<double> &weights) {
    double z = weights[0];
    for (int i=1; i<weights.size(); i++) {
        z += weights[i] * data[i-1];
    }
    return (z >= 0.0) ? 1 : 0;
}

vector<double> train(
    const vector<vector<double>> &trainSet, const double lr, const int numOfEpochs) {
    
    int numOfWeights = 1 + (trainSet[0].size() - 1);
    vector<double> weights(numOfWeights, 0);
    for (int e=0; e<numOfEpochs; e++) {
        double sumOfError = 0;
        for (auto data : trainSet) {
            double label = data[data.size() - 1];
            double pred = predict(data, weights);
            sumOfError += pow(label - pred, 2);
            
            // dL/db = dL/dy_hat * dy_hat/dz * dz/db
            weights[0] = weights[0] - lr * (pred - label);
            for (int i=1; i<weights.size(); i++) {
                weights[i] = weights[i] - lr * (pred - label) * data[i-1];
            }
        }
        cout << "epoch=" << e << ", sumOfError=" << sumOfError << endl;
    }
    
    return weights;
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
    
    // vector<double> weights = {-0.1, 0.20653640140000007, -0.23418117710000003};
    double lr = 0.1;
    int numOfEpochs = 5;
    vector<double> weights = train(trainSet, lr, numOfEpochs);
    
    // 
    cout << "weights:" << endl;
    for (auto val : weights) cout << val << endl;
    cout << "testing:" << endl;
    for (auto data : trainSet) {
        cout << predict(data, weights) << endl;
    }
}