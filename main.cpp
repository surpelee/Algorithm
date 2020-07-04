#include <iostream>
#include <vector>
#include <string>
#include "include/algorithm.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    Algorithm a;
    vector<int> nums = {-10,-3,0,1,5,9};
    vector<vector<int>> matrix = {{1,5,9},{10,11,13},{12,13,15}};
    string pattern = "bbbbbbbbbbbbbbabbbbb",value = "ppppppppppppppjsftcleifftfthiehjiheyqkhjfkyfckbtwbelfcgihlrfkrwireflijkjyppppg";
    auto ans = a.patternMatching(pattern,value);
    return 0;
}
