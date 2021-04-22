/*
CART (Classification and Regression Trees)

Regression:
The cost function that is minimized to choose split points is the sum squared error across all training samples that fall within the rectangle.

Classification:
The Gini cost function is used which provides an indication of how pure the node are, where node purity refers to how mixed the training data assigned to each node is.


1. Gini Index (cost function to evaluate splits in the dataset)
2. Create Split
3. Build a Tree
    3.1 Terminal Nodes (Maximum Tree Depth, Minimum Node Records)
    3.2 Recursive Splitting
    3.3 Building a Tree
4. Make a Prediction

*/

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <climits>
#include <cfloat>
#include <algorithm>

using namespace std;

double giniIndex(const vector<vector<vector<double>>> &groups, vector<double> classes) {
    
    // count all samples as split point
    int numOfInstances = 0;
    for (auto group : groups) {
        numOfInstances += group.size();
    }
    
    // sum weighted Gini index for each group
    double gini = 0;
    for (auto group : groups) {
        double size = group.size();
        if (size == 0) continue; // avoid divide by zero
        
        double score = 1.0; // 1 - p_1^2 - p_2^2 - ... - - p_N^2
        for (int classIdx : classes) {
            double count = 0;
            for (auto instance : group) {
                double label = instance[instance.size() - 1];
                if (label == classIdx) count++;
            }
            double p = count / size;
            score -= p * p;
        }
        gini += (size / numOfInstances) * score;
    }
    
    return gini;
}

vector<vector<vector<double>>> splitGroups(
    int featureIdx, double value, 
    const vector<vector<double>> &dataset) {
    
    vector<vector<double>> lefts;
    vector<vector<double>> rights;
    for (auto data : dataset) {
        if (data[featureIdx] < value) {
            lefts.push_back(data);
        } else {
            rights.push_back(data);
        }
    }
    
    return {lefts, rights};
}

struct Node {
    int featureIdx;
    double featureValue;
    double gini;
    vector<vector<vector<double>>> groups;
    
    Node* left = nullptr;
    Node* right = nullptr;
    double label = -1;
};

Node* getSplit(const vector<vector<double>> &dataset) {
    int numOfFeatures = dataset[0].size() - 1;
    
    // get labels lists
    unordered_set<double> bucket;
    for (auto data : dataset) {
        double classIdx = data[numOfFeatures];
        bucket.insert(classIdx);
    }
    vector<double> labels;
    for (auto label : bucket) labels.push_back(label);
    sort(labels.begin(), labels.end());
    
    // split groups by min gini
    double minGini = DBL_MAX;
    Node* info = new Node;
    for (int featureIdx=0; featureIdx<numOfFeatures; featureIdx++) {
        for (auto data : dataset) {
            auto groups = splitGroups(featureIdx, data[featureIdx], dataset);
            auto gini = giniIndex(groups, labels);
            // cout << "X1 < " << data[featureIdx] << ", gini = " << gini << endl;
            if (gini < minGini) {
                minGini = gini;
                info->featureIdx = featureIdx;
                info->featureValue = data[featureIdx];
                info->gini = gini;
                info->groups = groups;
            }
        }
    }
    return info;
}

// Create a terminal node value, and it will return most common output value
double toTerminal(const vector<vector<double>> &group) {
    unordered_map<double, int> counter;
    for (auto data : group) {
        double label = data[data.size()-1];
        if (counter.count(label) == 0) {
            counter[label] = 1;
        } else {
            counter[label] += 1;
        }
    }
    
    int maxCount = 0;
    double targetLabel;
    for (auto item : counter) {
        if (item.second > maxCount) {
            maxCount = item.second;
            targetLabel = item.first;
        }
    }
    return targetLabel;
}

// Create child splits for a node or make terminal
void split(Node* currNode, int maxDepth, int minSize, int depth) {
    auto leftGroup = currNode->groups[0];
    auto rightGroup = currNode->groups[1];
    currNode->groups.clear();
    
    // check for a no split
    if (leftGroup.empty() || rightGroup.empty()) {
        if (leftGroup.empty()) {
            currNode->right = new Node;
            currNode->right->label = toTerminal(rightGroup);
        } else {
            currNode->left = new Node;
            currNode->left->label = toTerminal(leftGroup);
        }
        return;
    }
    // check for max depth
    if (depth >= maxDepth) {
        currNode->left = new Node;
        currNode->left->label = toTerminal(leftGroup);
        currNode->right = new Node;
        currNode->right->label = toTerminal(rightGroup);
        return;
    }
    // process left child
    if (leftGroup.size() <= minSize) {
        currNode->left = new Node;
        currNode->left->label = toTerminal(leftGroup);
    } else {
        currNode->left = getSplit(leftGroup);
        split(currNode->left, maxDepth, minSize, depth+1);
    }
    // process right child
    if (rightGroup.size() <= minSize) {
        currNode->right = new Node;
        currNode->right->label = toTerminal(rightGroup);
    } else {
        currNode->right = getSplit(rightGroup);
        split(currNode->right, maxDepth, minSize, depth+1);
    }
}

Node* buildTree(
    const vector<vector<double>> &dataset, 
    int maxDepth, int minSize) {
    
    Node* root = getSplit(dataset);
    split(root, maxDepth, minSize, 1);
    return root;
}

void printTree(Node* root, int depth) {
    if (root == nullptr) return;
    
    if (root->label != -1) {
        cout << "depth: " << depth
            << ", label: " << root->label << endl;
    } else {
        cout << "depth: " << depth
            << ", featureIdx: " << root->featureIdx 
            << ", featureValue: " << root->featureValue << endl;
    }
    
    printTree(root->left, depth+1);
    printTree(root->right, depth+1);
}

double predict(Node* currNode, vector<double> data) {
    
    if (currNode->label != -1) return currNode->label;
    
    double featureValue = data[currNode->featureIdx];
    if (featureValue < currNode->featureValue) {
        if (currNode->left != nullptr) {
            return predict(currNode->left, data);
        }
    } else {
        if (currNode->right != nullptr) {
            return predict(currNode->right, data);
        }
    }
    return -1;
}

int main() {
    
    vector<vector<double>> dataset = {
        {2.771244718,1.784783929,0},
        {1.728571309,1.169761413,0},
        {3.678319846,2.81281357,0},
        {3.961043357,2.61995032,0},
        {2.999208922,2.209014212,0},
        {7.497545867,3.162953546,1},
        {9.00220326,3.339047188,1},
        {7.444542326,0.476683375,1},
        {10.12493903,3.234550982,1},
        {6.642287351,3.319983761,1}
    };
    
    Node* root = buildTree(dataset, 1, 1);
    
    printTree(root, 0);
    
    for (auto data : dataset) {
        double pred = predict(root, data);
        cout << "pred: " << pred << ", gt: " << data[data.size()-1] << endl;
    }
}