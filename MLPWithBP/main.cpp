/**
 * MLP with BackProb
 * 
 * 1. Initialize Weights
 * 2. Forward-Propagate
 * 3. Backpropage Error
 * 4. Train Network
 * 5. Predict
 * 
 */

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cassert>
#include <cmath>

using namespace std;

double getUniformRandom() {
    return 2 * ((double)rand() / RAND_MAX) - 1;
}

vector<vector<double>> getRandomWeights(int numOfInputs, int numOfOutputs) {
    vector<vector<double>> weights(numOfInputs, vector<double>(numOfOutputs));
    for (int i=0; i<numOfInputs; i++) {
        for (int j=0; j<numOfOutputs; j++) {
            weights[i][j] = getUniformRandom();
        }
    }
    return weights;
}

vector<double> getRandomBiases(int numOfOutputs) {
    vector<double> biases(numOfOutputs);
    for (int j=0; j<numOfOutputs; j++) {
        biases[j] = getUniformRandom();
    }
    return biases;
}

vector<double> sigmoid(vector<double> xs) {
    for (auto &x : xs) {
        x = 1.0 / (1.0 + exp(-x));
    }
    return xs;
}

vector<double> dSigmoid(vector<double> caches) {
    for (auto &cache : caches) {
        cache = cache * (1.0 - cache);
    }
    return caches;
}

double dSigmoid(double cache) {
    return cache * (1.0 - cache);
}

vector<double> linearCombine(
    const vector<vector<double>> &weights, 
    const vector<double> &biases,
    const vector<double> &inputs) {

    int numOfInputs = weights.size();
    int numOfOutputs = weights[0].size();
    assert(numOfInputs == inputs.size());

    vector<double> zs(numOfOutputs);
    for (int j=0; j<numOfOutputs; j++) {
        zs[j] = biases[j];
        for (int i=0; i<numOfInputs; i++) {
            zs[j] += weights[i][j] * inputs[i];
        }
    }

    return zs;
}

class Network {

public:
    Network(int, int, int);
    void printLayerWeights(string);
    void forwardPropagate(vector<double>);
    void backPropagate(vector<double>);
    void updateWeights(double learningRate);
    void train(const vector<vector<vector<double>>> &dataset, double learningRate, int numOfEpochs);
    void predict(vector<double>);

private:
    int mNumOfInputs;
    int mNumOfHiddens;
    int mNumOfOutputs;
    unordered_map<string, vector<vector<double>>> mWeights;
    unordered_map<string, vector<double>> mBiases;
    unordered_map<string, vector<double>> mCaches;
    unordered_map<string, vector<double>> mDeltas;
    void initWeightsAndBiases();
};

Network::Network(int numOfInputs, int numOfHidden, int numOfOutputs) {
    mNumOfInputs = numOfInputs;
    mNumOfHiddens = numOfHidden;
    mNumOfOutputs = numOfOutputs;
    initWeightsAndBiases();
}

void Network::initWeightsAndBiases() {
    // srand(time(NULL));

    mWeights["L1"] = getRandomWeights(mNumOfInputs, mNumOfHiddens);
    // mBiases["L1"] = getRandomBiases(mNumOfHiddens);
    mBiases["L1"] = vector<double>(mNumOfHiddens, 0);

    mWeights["L2"] = getRandomWeights(mNumOfHiddens, mNumOfOutputs);
    // mBiases["L2"] = getRandomBiases(mNumOfOutputs);
    mBiases["L2"] = vector<double>(mNumOfOutputs, 0);
}

void Network::printLayerWeights(string layerName) {
    if (mWeights.count(layerName) == 0) return;

    cout << layerName << "'s weights:" << endl;    
    for (auto row : mWeights[layerName]) {
        for (auto val : row) {
            cout << val << ", ";
        }
        cout << endl;
    }
    cout << layerName << "'s biases:" << endl;    
    for (auto val : mBiases[layerName]) {
        cout << val << ", ";
    }
    cout << endl;
}

void Network::forwardPropagate(vector<double> inputs) {
    vector<double> outputs = inputs;

    mCaches["L0"] = outputs;
    for (string layerName : {"L1", "L2"}) {
        outputs = linearCombine(mWeights[layerName], mBiases[layerName], outputs);
        outputs = sigmoid(outputs);
        mCaches[layerName] = outputs;
    }
}

string getPrevLayerName(string layerName) {
    string oldKey = layerName.substr(1, layerName.size() - 1);
    string newKey = to_string(stoi(oldKey) - 1);
    return "L" + newKey;
}

string getNextLayerName(string layerName) {
    string oldKey = layerName.substr(1, layerName.size() - 1);
    string newKey = to_string(stoi(oldKey) + 1);
    return "L" + newKey;
}

void Network::backPropagate(vector<double> gts) {
    for (string layerName : {"L2", "L1"}) {
        int numOfOutputs = mBiases[layerName].size();
        mDeltas[layerName] = vector<double>(numOfOutputs, 0);

        if (layerName == "L2") {
            // deltas = diff * dSigmoid
            for (int i=0; i<numOfOutputs; i++) {
                mDeltas[layerName][i] = (mCaches[layerName][i] - gts[i]) * dSigmoid(mCaches[layerName][i]);
            }
        } else {
            // deltas = nextDeltas * nextWeight * dSigmoid
            string nextLayerName = getNextLayerName(layerName);
            int numOfNextOutputs = mBiases[nextLayerName].size();
            for (int i=0; i<numOfOutputs; i++) {
                for (int j=0; j<numOfNextOutputs; j++) {
                    mDeltas[layerName][i] += mDeltas[nextLayerName][j] * mWeights[nextLayerName][i][j];
                }
                mDeltas[layerName][i] *= dSigmoid(mCaches[layerName][i]);
            }
        }
    }
}

void Network::updateWeights(double learningRate) {
    for (string layerName : {"L1", "L2"}) {
        for (int i=0; i<mBiases[layerName].size(); i++) {
            mBiases[layerName][i] -= learningRate * mDeltas[layerName][i];
        }
        for (int i=0; i<mWeights[layerName].size(); i++) {
            for (int j=0; j<mWeights[layerName][0].size(); j++) {
                mWeights[layerName][i][j] -= learningRate * mDeltas[layerName][j] * mCaches[getPrevLayerName(layerName)][i];
            }
        }
    }
}

void Network::train(
    const vector<vector<vector<double>>> &dataset, double learningRate, int numOfEpochs) {

    for (int e=0; e<numOfEpochs; e++) {
        for (auto data : dataset) {
            forwardPropagate(data[0]);
            backPropagate(data[1]);
            updateWeights(learningRate);

            cout << data[1][0] << ":" << mCaches["L2"][0] << endl;
        }
        
        cout << "epoch: " << e << endl;
        //printLayerWeights("L2");
    }
}

int main () {
    
    Network net(2, 3, 1);


    for (auto layerName : {"L1", "L2"}) {
        net.printLayerWeights(layerName);
    }

    vector<vector<vector<double>>> dataset = {
        {{15, 11}, {0}},
        {{11, 19}, {0}},
        {{7, 15}, {0}},
        {{11, 9}, {0}},
        {{-1, -2}, {1}},
        {{-5, -12}, {1}},
        {{-4, -1}, {1}},
        {{-1, -11}, {1}},
    };
    net.train(dataset, 0.5, 20);
}