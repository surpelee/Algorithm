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
#include <math.h>
#include <algorithm>

using namespace std;

struct ListNode{
    int val;
    ListNode* next;
    ListNode(int _val):val(_val),next(nullptr){}
};

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
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid);//不同路径2
    bool hasPathSum(TreeNode* root, int sum);//路径总和
    vector<int> divingBoard(int shorter, int longer, int k);//跳水板
    int respace(vector<string>& dictionary, string sentence);//恢复空格
    int maxProfit_freeze(vector<int>& prices);//买卖股票最佳时机包含冷冻期
    int maxProfit(vector<int>& prices);//买卖股票最佳时机
    int maxProfit2(vector<int>& prices);//买卖股票最佳时机 多次买卖股票
    int maxProfit3(vector<int>& prices);//买卖股票最佳时机 最多两次买卖股票
    int maxProfit4(int k,vector<int>& prices);//买卖股票最佳时机 最多k次买卖股票
    vector<int> countSmaller(vector<int>& nums);//计算右侧小于当前元素的个数
    int calculateMinimumHP(vector<vector<int>>& dungeon);//地下城游戏
    int numIdenticalPairs(vector<int>& nums);//好数对的个数
    int numSub(string s);//仅含 1 的子串数
    double maxProbability(int n, vector<vector<int>>& edges, vector<double>& succProb, int start, int end);//概率最大的路径
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2);//两个数组的交集
    int minimumTotal(vector<vector<int>>& triangle);//三角形最小路径和
    int numTree(int n);//不同的二叉搜索树
    bool isBipartite(vector<vector<int>>& graph);//判断二分图
    int searchInsert(vector<int>& nums, int target);//搜索插入位置
    vector<TreeNode*> generateTrees(int n);//不同的二叉搜索树2
    int minArray(vector<int>& numbers);//旋转数组的最小数字
    int minPathSum(vector<vector<int>>& grid);//最小路径和
    bool divisorGame(int N);//除数博弈
    bool isSubsequence(string s, string t);//判断子序列
    int maxDepth(TreeNode* root);//二叉树的最大深度
    int integerBreak(int n);//整数拆分
    int findMagicIndex(vector<int>& nums);//魔术索引

private:
    int back_findMagicIndex(vector<int>& nums,int l,int r);//魔术索引
    vector<TreeNode*> back_generateTrees(int l,int r);
    void back_respace(unordered_set<string> &dictionary, string &sentence, int wordLen, int x,int num);
    bool back_hasPathSum(TreeNode *node, int sum, int ans);
    void CountPalin(const string& s,int l,int r);
    bool check_searchMatrix(vector<vector<int>>& matrix,int mid,int i,int j);
    TreeNode* back_sortedArrayToBST(vector<int>& nums,int l,int r);

    //构建堆
    void maxHeapify(vector<int>& a,int i,int heapSize);
    void buildMaxHeap(vector<int>& a,int heapSize);
private:
    int m_int;
    vector<int> m_intVt;

    void mergeCountSmaller(vector<int> &nums, vector<int> &index, vector<int> &res, int l, int r);

public:
    bool KMPString(const string& s,const string& t);
    ListNode* reverseList(ListNode* head);
    vector<int> preorderTraversal(TreeNode* root);
    void back_Preorder(TreeNode* root);
    vector<int> inorderTraversal(TreeNode* root);


};

#endif