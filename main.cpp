#include <iostream>
#include <vector>
#include <string>
#include "include/algorithm.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    Algorithm a;
    vector<int> p = {5,2,6,1};
    vector<string> s = {"jxnonurhhuanyuqahjy","phrxu","hjunypnyhajaaqhxduu"};
    vector<vector<int>> dun = {{1},{0,3},{3},{1,2}};
    vector<double> pp = {0.06,0.26,0.49,0.25,0.2,0.64,0.23,0.21,0.77};
    string tmp = "000000";
    auto ans = a.isBipartite(dun);
    return 0;
}
