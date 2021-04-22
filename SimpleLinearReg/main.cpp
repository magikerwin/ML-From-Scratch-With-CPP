/*
Simple linear regression algorithm:

1. calculate mean and variance
2. calculate covariance
3. estimate coefficients (b1=cov(x,y)/var(x), b0=mean(y)-b1*mean(x))
4. make predicitons

Note: 
sum(y_i - y_mean) = sum(y_i) - sum(y_mean) = 0
sum((x_i - x_mean) * (y_i * y_mean)) = sum(x_i * (y_i * y_mean))
*/

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

double getMean(vector<double> values) {
    double sum = 0;
    for (auto value : values) {
        sum += value;
    }
    return sum / values.size();
}

double getVariance(vector<double> values) {
    double mean = getMean(values);
    double variance = 0;
    for (auto value : values) {
        variance += pow(value - mean, 2);
    }
    return variance;
}

double getCovariance(vector<double> valuesA, vector<double> valuesB) {
    double meanA = getMean(valuesA);
    double meanB = getMean(valuesB);
    double covariance = 0;
    for (int i=0; i<valuesA.size(); i++) {
        covariance += (valuesA[i] - meanA) * (valuesB[i] - meanB);
    }
    return covariance;
}

vector<double> getCoefficients(const vector<vector<double>> &dataset) {
    // get xs and ys
    vector<double> xs;
    vector<double> ys;
    for (auto pair : dataset) {
        xs.push_back(pair[0]);
        ys.push_back(pair[1]);
    }
    
    // calculate coefficients
    double b1 = getCovariance(xs, ys) / getVariance(xs);
    double b0 = getMean(ys) - b1 * getMean(xs);
    
    return {b0, b1};
}

vector<double> myGetCoefficients(const vector<vector<double>> &dataset) {
    // get xs and ys
    vector<double> xs;
    vector<double> ys;
    for (auto pair : dataset) {
        xs.push_back(pair[0]);
        ys.push_back(pair[1]);
    }
    
    // calculate coefficients
    double numeratro = 0;
    for (int i=0; i<dataset.size(); i++) {
        numeratro += dataset[i][0] * (ys[i] - getMean(ys));
    }
    double denuminator = 0;
    for (int i=0; i<dataset.size(); i++) {
        denuminator += dataset[i][0] * (xs[i] - getMean(xs));
    }
    
    double b1 = numeratro / denuminator;
    double b0 = getMean(ys) - b1 * getMean(xs);
    
    return {b0, b1};
}

double predict(double x, vector<double> coefficents) {
    return x * coefficents[1] + coefficents[0];
}

double evaluation(
    const vector<vector<double>> &dataset, 
    const vector<double> &predictions) {
    
    // rmse
    double sum = 0;
    for (int i=0; i<dataset.size(); i++) {
        sum += pow(dataset[i][1] - predictions[i], 2);
    }
    return sqrt(sum / dataset.size());
}

int main() {
    
    vector<vector<double>> trainSet = {
        {1, 1},
        {2, 3},
        {4, 3},
        {3, 2},
        {5, 5}
    };
    
    auto coefficents = getCoefficients(trainSet);    
    cout << "coefficents = {" 
        << coefficents[0] << ", "
        << coefficents[1] << "}"<< endl;
    vector<double> predictions;
    for (auto pair : trainSet) {
        predictions.push_back(predict(pair[0], coefficents));
    }
    double rms = evaluation(trainSet, predictions);
    cout << rms << endl;
}