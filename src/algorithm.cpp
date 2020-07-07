#include <iostream>
#include "../include/algorithm.h"


using namespace std;

int Algorithm::threeSumClosest(vector<int> &nums, int target) {
    std::sort(nums.begin(),nums.end());
    int res = nums[0] + nums[1] + nums[2];
    for(int i = 0;i<nums.size()-2;++i){
        if(i>0&&nums[i] == nums[i-1]) continue;
        int l = i + 1,r = nums.size() - 1;
        while (l<r){
            int sum = nums[i] + nums[l] + nums[r];
            if(abs(target - res) > abs(target - sum))
                res = sum;
            if(sum > target){
                r --;
            }else if(sum < target){
                l ++;
            }else{
                return sum;
            }
        }
    }
    return res;
}

int Algorithm::countSubstrings(string s) {
    /*int sSize = s.size();
    m_int = 0;
    for(int i = 0;i<sSize;++i){
        CountPalin(s,i,i);
        CountPalin(s,i,i+1);
    }
    return m_int;*/
    int sSize = s.size();
    m_int = sSize;
    vector<vector<bool>> dp(sSize,vector<bool>(sSize,false));
    for(int i = 0;i<s.size();++i) dp[i][i] = true;
    for(int i = sSize - 1;i >= 0;--i){
        for(int j = i + 1;j<sSize;++j){
            if(s[i] == s[j]){
                dp[i][j] = j - i == 1 ? true : dp[i+1][j-1];
            } else{
                dp[i][j] = false;
            }
            if(dp[i][j]) m_int++;
        }
    }
    return m_int;
}

void Algorithm::CountPalin(const string& s, int l, int r) {
    while (l<=r&&l>=0&&r<s.size()&&s[l] == s[r]){
        m_int++;
        l--;
        r++;
    }
}

int Algorithm::minSubArrayLen(int s, vector<int> &nums) {
    int l = 0,r = 0;
    int res = 0,ans = nums.size() + 1;
    for(;r<nums.size();++r){
        res += nums[r];
        while(res>=s){
            ans = min(ans,r - l + 1);
            res -= nums[l];
            l++;
        }
    }
    return ans == nums.size() + 1 ? 0 : ans;
}

int Algorithm::findKthLargest(vector<int> &nums, int k) {
    /*priority_queue<int,vector<int>,greater<int> > q;
    for(int i = 0;i<nums.size();++i){
        if(q.size()<k){
            q.push(nums[i]);
        }
        else if(q.top() < nums[i]){
            q.pop();
            q.push(nums[i]);
        }
    }
    return q.top();*/
    //手动实现最大堆
    int heapSize = nums.size();
    buildMaxHeap(nums,heapSize);
    for(int i = nums.size() - 1;i >= nums.size() - k + 1;--i){
        swap(nums[0],nums[i]);
        --heapSize;
        maxHeapify(nums,0,heapSize);
    }
    return nums[0];
}

void Algorithm::maxHeapify(vector<int> &a, int i, int heapSize) {
    int l = i * 2 + 1,r = i * 2 + 2,largest = i;
    if(l < heapSize && a[l] > a[largest])
        largest = l;
    if(r < heapSize && a[r] > a[largest])
        largest = r;
    if(largest != i){
        swap(a[largest],a[i]);
        maxHeapify(a,largest,heapSize);
    }
}

void Algorithm::buildMaxHeap(vector<int> &a, int heapSize) {
    for(int i = heapSize / 2;i >= 0;--i){
        maxHeapify(a,i,heapSize);
    }
}

int Algorithm::findLength(vector<int> &A, vector<int> &B) {
    int ans = 0;
    int m = A.size(),n = B.size();
    vector<vector<int>> dp(m,vector<int>(n,0));
    for(int i = 1;i<=m;++i){
        for(int j = 1;j<=n;++j){
            if(A[i - 1] == B[j - 1]){
                dp[i][j] = dp[i - 1][j - 1] + 1;
                ans = max(ans,dp[i][j]);
            }
        }
    }
    return ans;
}

int Algorithm::kthSmallest(vector<vector<int>> &matrix, int k) {
    //最小堆
    /*int n = matrix.size();
    struct node{
        int x,y,val;
        node(int _x,int _y,int _val):x(_x),y(_y),val(_val) {}
        bool operator > (const node& a) const {
            return a.val < this->val;
        }
    };
    priority_queue<node,vector<node>,greater<node> > q;
    for(int i = 0;i<n;++i){
        q.push(node(i,0,matrix[i][0]));
    }
    for(int i = 0;i< k - 1;++i){
        node now = q.top();
        q.pop();
        if(now.y != n - 1)
            q.push(node(now.x,now.y + 1,matrix[now.x][now.y + 1]));
    }
    return q.top().val;*/
    //二分查找
    int n = matrix.size();
    int l = matrix[0][0],r = matrix[n - 1][n - 1];
    while (l<=r){
        int mid = l + (r - l)/2;
        if(check_searchMatrix(matrix,mid,k,n)){
            r = mid;
        } else{
            l = mid + 1;
        }
    }
    return l;
}

bool Algorithm::searchMatrix(vector<vector<int>> &matrix, int target) {
    if(matrix.size() == 0) return false;
    int m = matrix.size(),n = matrix[0].size();
    int i = m - 1,j = 0;
    while(i>=0&&j<n){
        int mid = matrix[i][j];
        if(mid == target) return true;
        else if(mid < target) --i;
        else ++j;
    }
    return false;
}

bool Algorithm::check_searchMatrix(vector<vector<int>> &matrix, int mid, int k, int n) {
    int num = 0;
    int i = n - 1,j = 0;
    while(i>=0&&j<matrix.size()){
        if(matrix[i][j] <= mid){
            num += i + 1;
            ++j;
        }else{
            --i;
        }
    }
    return num >= k;
}

TreeNode *Algorithm::sortedArrayToBST(vector<int> &nums) {
    return back_sortedArrayToBST(nums,0,nums.size() - 1);
}

TreeNode *Algorithm::back_sortedArrayToBST(vector<int> &nums, int l, int r) {
    if(l>r){
        return nullptr;
    }
    int mid = (r + l + 1)/2;
    TreeNode* node = new TreeNode(nums[mid]);
    node->left = back_sortedArrayToBST(nums,l,mid - 1);
    node->right = back_sortedArrayToBST(nums,mid + 1,r);
    return node;
}

bool Algorithm::patternMatching(string pattern, string value) {
    string a = "",b = "";
    int aSize = 0,bSize = 0;
    int n = value.size();
    for(const auto& s:pattern){
        if(s == 'a') aSize++;
        else bSize++;
    }
    if(aSize + bSize == 1&&value.empty()) return true;
    if(aSize == 0&&bSize == 0){
        return 0;
    }else if(aSize == 0){
        int tmp = n/bSize;
        string temp_b = value.substr(0,tmp);
        for(int k = 0;k<n;k += tmp){
            if(value.substr(k,tmp) != temp_b) return false;
        }
        return true;
    }else if(bSize == 0){
        int tmp = n/aSize;
        string temp_a = value.substr(0,tmp);
        for(int k = 0;k<n;k += tmp){
            if(value.substr(k,tmp) != temp_a) return false;
        }
        return true;
    }

    for(int i = 0;i<n/aSize;++i){
        int j = (n - aSize * i) / bSize;
        if(aSize*i + bSize*j != n) continue;
        a = value.substr(0,i);
        b = value.substr(i,j);
        if(a == b) continue;
        int pos = 0;
        for(int k = 0;k<pattern.size()&&pos<n;++k){
            if(pattern[k] == 'a' && value.substr(pos,i) == a) {
                pos += i;
                continue;
            }
            else if(value.substr(pos,j) == b) {
                pos += j;
                continue;
            }
            else break;
        }
        if(pos == n) return true;
    }

    //reverse(pattern.begin(),pattern.end());
    //reverse(value.begin(),value.end());

    for(int i = 0;i<n/bSize;++i){
        int j = (n - bSize * i) / aSize;
        if(bSize*i + aSize*j != n) continue;
        b = value.substr(0,i);
        a = value.substr(i,j);
        if(a == b) continue;
        int pos = 0;
        for(int k = 0;k<pattern.size()&&pos<n;++k){
            if(pattern[k] == 'b' && value.substr(pos,i) == b) {
                pos += i;
                continue;
            }
            else if(value.substr(pos,j) == a) {
                pos += j;
                continue;
            }
            else break;
        }
        if(pos == n) return true;
    }

    return false;
}

int Algorithm::longestValidParentheses(string s) {
    stack<int> si;
    si.push(-1);
    int ans = 0;
    for(int i = 0;i<s.size();++i){
        if(si.top() == -1){
            si.push(i);
        }
        else if(s[i] == ')'&&s[si.top()] == '('){
            si.pop();
            ans = max(ans,i - si.top());
        }else
            si.push(i);
    }
    return ans;
}

int Algorithm::maxScoreSightseeingPair(vector<int> &A) {
    int ans = 0,tmp = A[0];
    for(int i = 1;i<A.size();++i){
        ans = max(ans,tmp + A[i] - i);
        tmp = max(tmp,A[i] + i);
    }
    return ans;
}

int Algorithm::uniquePathsWithObstacles(vector<vector<int>> &obstacleGrid) {
    int m = obstacleGrid.size();
    int n = obstacleGrid[0].size();
    vector<vector<int>> dp(m,vector<int>(n,0));
    for(int i = 0;i<m;++i){
        if(obstacleGrid[i][0] == 1)
            break;
        dp[i][0] = 1;
    }
    for(int i = 0;i<n;++i){
        if(obstacleGrid[0][i] == 1)
            break;
        dp[0][i] = 1;
    }
    for(int i = 1;i<m;++i){
        for(int j = 1;j<n;++j){
            if(obstacleGrid[i][j] == 1){
                continue;
            }
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }
    return dp[m - 1][n - 1];
}

bool Algorithm::hasPathSum(TreeNode *root, int sum) {
    return back_hasPathSum(root,sum,0);
}

bool Algorithm::back_hasPathSum(TreeNode *node,int sum,int ans) {
    if(node == nullptr)
        return false;
    ans += node->val;
    if(!node->left&&!node->right&&ans == sum) return true;
    if(back_hasPathSum(node->left,sum,ans) || back_hasPathSum(node->right,sum,ans))
        return true;
    ans -= node->val;
    return false;
}

























