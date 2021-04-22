/*
knn algorithm:

1. get neighbors by euclidean distance
2. make predictions according to neigbors (classification: argmax, regression: mean)
*/

#include <iostream>
#include <vector>
#include <unordered_map>
#include <climits>
#include <cmath>
#include <algorithm>

using namespace std;

double euclideanDistance(vector<double> data1, vector<double> data2) {
    double distance = 0;
    for (int i=0; i<data1.size(); i++) {
        distance += pow(data1[i] - data2[i], 2);
    }
    return sqrt(distance);
}

vector<int> getNeighbors(
    const vector<vector<double>> &trainSet, 
    const vector<double> testData, int numOfNeighbors) {
    
    // calculate distance
    vector<pair<double, int>> distIdxPairs(trainSet.size());
    for (int i=0; i<trainSet.size(); i++) {
        distIdxPairs[i].first = euclideanDistance(trainSet[i], testData);
        distIdxPairs[i].second = i;
    }    
    
    // sorting by distance
    sort(distIdxPairs.begin(), distIdxPairs.end());
    
    // get indexes of k nearest neighbors
    vector<int> neighbors(numOfNeighbors);
    for (int i=0; i<numOfNeighbors; i++) {
        neighbors[i] = distIdxPairs[i].second;
    }
    
    return neighbors;
}

double predict(
    const vector<vector<double>> &trainSet, 
    const vector<double> testData, int numOfNeighbors) {
    
    // get k nearest neighbors
    auto neighbors = getNeighbors(trainSet, testData, numOfNeighbors);
    
    // count frequency
    unordered_map<double, int> counter;
    for (int neighbor : neighbors) {
        double label = trainSet[neighbor][2];
        if (counter.count(label) == 0) {
            counter[label] = 0;
        } else {
            counter[label]++;
        }
    }
    
    // find max count
    int maxCount = INT_MIN;
    int prediction = -1;
    for (auto item : counter) {
        if (item.second > maxCount) {
            maxCount = item.second;
            prediction = item.first;
        }
    }
    
    return prediction;
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
    
    double prediciton = predict(trainSet, trainSet[0], 3);
    cout << "input: {" 
        << trainSet[0][0] << ", " << trainSet[0][1] 
        << "}, prediction: " << prediciton << endl;
    
}