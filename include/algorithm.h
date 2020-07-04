#ifndef _ALGORITHM_ALGORITHM_H_
#define _ALGORITHM_ALGORITHM_H_

#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <stack>
#include <list>
#include <functional>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>

using namespace std;

struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int _val):val(_val),left(nullptr),right(nullptr){}
};

//用两个栈实现队列
class CQueue {
/*public:
    CQueue() {
        while(!s1.empty())
            s1.pop();
        while(!s2.empty())
            s2.pop();
    }

    void appendTail(int value) {
        s1.push(value);
    }

    int deleteHead() {
        if(s2.empty()){
            while(!s1.empty()){
                int temp = s1.top();
                s1.pop();
                s2.push(temp);
            }
        }
        if(s2.empty()) return -1;
        int ans = s2.top();
        s2.pop();
        return ans;
    }

private:
    stack<int> s1,s2;*/
public:
    CQueue(){
        s1.clear();
    }

    void appendTail(int value){
        s1.push_back(value);
    }

    int deleteHead(){
        if(s1.empty()) return -1;
        int temp = s1.front();
        s1.pop_front();
        return temp;
    }

private:
    list<int> s1;
};

class Algorithm
{
public:
    Algorithm(){}
    ~Algorithm(){}

    int threeSumClosest(vector<int>& nums, int target);
    int countSubstrings(string s);
    int minSubArrayLen(int s, vector<int>& nums);
    int findKthLargest(vector<int>& nums, int k);
    int findLength(vector<int>& A, vector<int>& B);
    int kthSmallest(vector<vector<int>>& matrix, int k);//有序矩阵中第K小的元素
    bool searchMatrix(vector<vector<int>>& matrix, int target);//搜索二维矩阵2
    TreeNode* sortedArrayToBST(vector<int>& nums);//将有序数组转换为二叉树
    bool patternMatching(string pattern, string value);//模式匹配
    int longestValidParentheses(string s);//最长有效括号
    int maxScoreSightseeingPair(vector<int>& A);//最佳观光组合

private:
    void CountPalin(const string& s,int l,int r);
    bool check_searchMatrix(vector<vector<int>>& matrix,int mid,int i,int j);
    TreeNode* back_sortedArrayToBST(vector<int>& nums,int l,int r);

    //构建堆
    void maxHeapify(vector<int>& a,int i,int heapSize);
    void buildMaxHeap(vector<int>& a,int heapSize);
private:
    int m_int;

};

#endif