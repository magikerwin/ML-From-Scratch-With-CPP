/* 
k-means clustering is the task of finding groups of 
pointrs in a dataset such that the total variance within
groups is minimized.
 
--> find argmin(sum(xi - ci)^2)

algorithm:

1. init the clusters

iterations {
    2. assign each point to the nearest centroid
    3. redefine the cluster
}

*/

#include <ctime>    // for a random seed
#include <fstream>  // for file reading
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>    // for pow()
#include <cfloat>   // for DBL_MAX

using namespace std;

struct Point {
    int x, y;
    int cluster;
    double minDistance;

    Point(int _x, int _y) {
        x = _x;
        y = _y;
        cluster = -1;
        minDistance = DBL_MAX;
    }

    double distance(Point p) {
        return pow((this->x - p.x), 2) + pow(this->y - p.y, 2);
    }
};

vector<Point> readCSV(string path) {
    vector<Point> points;
    string line;
    ifstream file(path);

    getline(file, line); // pop header
    while (getline(file, line)) {
        stringstream lineStream(line);

        double x, y;
        string bit;
        getline(lineStream, bit, ',');
        getline(lineStream, bit, ',');
        getline(lineStream, bit, ',');
        getline(lineStream, bit, ',');
        x = stof(bit);
        getline(lineStream, bit, '\n');
        y = stof(bit);
        
        points.push_back(Point(x, y));
    }

    file.close();
    return points;
}

void kMeansClustering(vector<Point> &points, int epochs, int k) {
    
    // 1. init centroids
    vector<Point> centroids;
    srand(time(0)); // need to set the random seed
    int numOfPoints = points.size();
    for (int i=0; i<k; i++) {
        //int pointIdx = rand() % numOfPoints;
        int pointIdx = i;
        centroids.push_back(points.at(pointIdx));
        centroids.back().cluster = i;
    }

    // do some iterations
    for (int e=0; e<epochs; e++) {

        // 2. assign points to a cluster
        for (auto &point : points) {
            point.minDistance = DBL_MAX;
            for (int c=0; c<centroids.size(); c++) {
                double distance = point.distance(centroids[c]);
                if (distance < point.minDistance) {
                    point.minDistance = distance;
                    point.cluster = c;
                }
            }
        }

        // 3. redefine centroids
        vector<int> sizeOfEachCluster(k, 0);
        vector<double> sumXOfEachCluster(k, 0);
        vector<double> sumYOfEachCluster(k, 0);
        for (auto point : points) {
            sizeOfEachCluster[point.cluster] += 1;
            sumXOfEachCluster[point.cluster] += point.x;
            sumYOfEachCluster[point.cluster] += point.y;
        }
        for (int i=0; i<centroids.size(); i++) {
            centroids[i].x = (sizeOfEachCluster[i] == 0) ? 0 : sumXOfEachCluster[i] / sizeOfEachCluster[i];
            centroids[i].y = (sizeOfEachCluster[i] == 0) ? 0 : sumYOfEachCluster[i] / sizeOfEachCluster[i];
        }

        // 4. write to a file
        ofstream file1;
        file1.open("points_iter_" + to_string(e) + ".csv");
        file1 << "x,y,clusterIdx" << endl;
        for (auto point : points) {
            file1 << point.x << "," << point.y << "," << point.cluster << endl;
        }
        file1.close();
        
        ofstream file2;
        file2.open("centroids_iter_" + to_string(e) + ".csv");
        file2 << "x,y,clusterIdx" << endl;
        for (auto centroid : centroids) {
            file2 << centroid.x << "," << centroid.y << "," << centroid.cluster << endl;
        }
        file2.close();

    }
    
}

int main() {
    // [option 1] load csv
    vector<Point> points = readCSV("./mall_customers.csv");
    // [option 2] 
    // vector<Point> points = {
    //     {12, 39}, {20, 36}, {28, 30}, {18, 52}, {29, 54}, {33, 46}, {24, 55}, {45, 59}, {60, 35}, {52, 70},
    //     {51, 66}, {52, 63}, {55, 58}, {53, 23}, {55, 58}, {53, 23}, {55, 14}, {61, 8}, {64, 19}, {69, 7}, {72, 24}
    // };

    kMeansClustering(points, 5, 6);
}