#include <iostream>
#include <vector>
#include <string>
#include "include/algorithm.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    Algorithm a;
    vector<int> p = {5,2,6,1};
    vector<string> s = {"jxnonurhhuanyuqahjy","phrxu","hjunypnyhajaaqhxduu"};
    vector<vector<int>> dun = {{2,3},{1,2},{3,4},{1,3},{1,4},{0,1},{2,4},{0,4},{0,2}};
    vector<double> pp = {0.06,0.26,0.49,0.25,0.2,0.64,0.23,0.21,0.77};
    string tmp = "000000";
    auto ans = a.maxProbability(5,dun,pp,0,3);
    return 0;
}
