/*
Adam algorithm

1. init beta1(0.9), beta2(0.999), epsilon(1e-8)
2. calculate moving average for momentum and RMS
3. bias correction for moving average
4. update weights

*/

#include <iostream>
#include <cmath>

using namespace std;

class SGDOptimizer {
public:
    SGDOptimizer(float learningRate=0.01, float momentum=0.9) {
        // 1. init
        mLearningRate = learningRate;
        mMomentum = momentum;
        mMovingAvg = 0;
    }

    void update(int t, float &weight, float dWeight) {
        // 2. calculate moving averate
        mMovingAvg = mMomentum * mMovingAvg + (1 - mMovingAvg) * dWeight;
        // 3. bias correction
        float correctedMovingAvg = mMovingAvg / (1 - pow(mMomentum, t));
        // 4. update weights
        weight -= mLearningRate * correctedMovingAvg;
    }

private:
    float mLearningRate;
    float mMomentum;
    float mMovingAvg;
};

class AdamOptimizer {
public:
    AdamOptimizer(float learningRate=0.01, float beta1=0.9, float beta2=0.999, float epsilon=1e-8) {
        
        // 1. init beta1(0.9), beta2(0.999), epsilon(1e-8)
        mLearningRate = learningRate;
        mBeta1 = beta1;
        mBeta2 = beta2;
        mEpsilon = epsilon;

        mMovingAvg = 0;
        mMovingAvgOfRMS = 0;
    }

    void update(int t, float &weight, float dWeight) {
        // 2. calculate moving average for momentum and RMS
        mMovingAvg = mBeta1 * mMovingAvg + (1 - mBeta1) * dWeight;
        mMovingAvgOfRMS = mBeta2 * mMovingAvgOfRMS + (1 - mBeta2) * (dWeight * dWeight);

        // 3. bias correction for moving average
        float correctedMovingAvg = mMovingAvg / (1 - pow(mBeta1, t)); // t start from 1
        float correctedMovingAvgOfRMS = mMovingAvgOfRMS / (1 - pow(mBeta2, t));

        // 4. update weights
        weight -= mLearningRate * correctedMovingAvg / (sqrt(correctedMovingAvgOfRMS) + mEpsilon);
    }

private:
    float mLearningRate;
    float mBeta1, mBeta2;
    float mEpsilon;
    float mMovingAvg;
    float mMovingAvgOfRMS;
};

float getLoss(float x) {
    return x*x - 2*x + 1;
}

float getGradient(float x) {
    return 2*x - 2;
}

int main() {
    float weight = 0;
    int t = 1;
    AdamOptimizer opt(0.01);
    // SGDOptimizer opt(0.01);

    while (true) {
        float dWeight = getGradient(weight);

        float previousWeight = weight;
        opt.update(t, weight, dWeight);
        if (previousWeight == weight) {
            cout << "converged!" << endl;
            break;
        } else {
            cout << "iter: " << t << ", weight = " << weight << endl;
            t++;
        }

        if (t > 10000) break;
    }
}