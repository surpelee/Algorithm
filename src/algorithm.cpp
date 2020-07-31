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

vector<int> Algorithm::divingBoard(int shorter, int longer, int k) {
    if(shorter == longer){
        vector<int> ans;
        if(k == 0) return ans;
        ans.push_back(longer*k);
        return  ans;
    }
    vector<int> ans(k + 1);
    for(int i = 0;i <= k;++i){
        int tmp = i*longer + shorter*(k - i);
        ans[i] = tmp;
    }
    return ans;
}

int Algorithm::respace(vector<string> &dictionary, string sentence) {
    //递归失败
    /*unordered_set<string> aSet;
    int wordLen = 0;
    for(const auto& s:dictionary) {
        wordLen = max(wordLen,(int)s.size());
        aSet.insert(s);
    }
    m_int = sentence.size();
    back_respace(aSet,sentence,wordLen,0,0);
    return m_int;*/
    //动态规划
    int n=sentence.size();
    int dp[n+1];
    dp[0]=0;
    for(int i=0;i<n;++i){
        dp[i+1]=dp[i]+1;
        for(auto& word:dictionary){
            if(word.size()<=i+1){
                if(word==sentence.substr(i+1-word.size(),word.size()))
                    dp[i+1]=min(dp[i+1],dp[i+1-word.size()]);
            }
        }
    }
    return dp[n];
}

void Algorithm::back_respace(unordered_set<string>& dictionary,string& sentence,int wordLen,int x,int num){
    if(x >= sentence.size()){
        m_int = min(m_int,num);
        return;
    }
    for(int i = x + 1;i<=sentence.size();++i){
        if(i - x > wordLen) {
            back_respace(dictionary,sentence,wordLen,x + 1,num + 1);
            return;
        }
        string tmp = sentence.substr(x,i - x);
        if(dictionary.find(tmp) == dictionary.end()){
            continue;
        }
        back_respace(dictionary,sentence,wordLen,i,num);
    }
}

int Algorithm::maxProfit_freeze(vector<int> &prices) {
    if(prices.empty()) return 0;
    int sSize = prices.size();
    vector<vector<int>> dp(sSize,vector<int>(2,0));
    dp[0][0] = 0;
    dp[0][1] = -prices[0];
    int dp_pre = 0;//代表dp[i-2][0]
    for(int i = 1;i<sSize;++i){
        int tmp = dp[i-1][0];
        dp[i][0] = max(dp[i-1][0],dp[i-1][1] + prices[i]);
        dp[i][1] = max(dp[i-1][1],dp_pre - prices[i]);
        dp_pre = tmp;
    }
    return dp[sSize - 1][0];
}

int Algorithm::maxProfit(vector<int> &prices) {
    if(prices.empty()) return 0;
    int iMin = INT_MAX,iMax = 0;
    for(int i = 0;i<prices.size();++i){
        iMax = max(iMax,prices[i] - iMin);
        iMin = min(iMin,prices[i]);
    }
    return iMax;
    /*if (prices.empty()) return 0;
    int ans(0),profit(0),pre = prices[0];
    for (int i = 1; i < prices.size(); ++i) {
        if (pre > prices[i]) pre = prices[i];
        else  profit = prices[i] - pre;
        ans = ans > profit ? ans : profit;
    }
    return ans;*/
}

int Algorithm::maxProfit2(vector<int> &prices) {
    if(prices.empty()) return 0;
    int sSize = prices.size();
    vector<vector<int>> dp(sSize,vector<int>(2,0));
    dp[0][0] = 0;
    dp[0][1] = -prices[0];
    for(int i = 1;i<sSize;++i){
        dp[i][0] = max(dp[i-1][0],dp[i-1][1] + prices[i]);
        dp[i][1] = max(dp[i-1][1],dp[i-1][0] - prices[i]);
    }
    return dp[sSize - 1][0];
}

int Algorithm::maxProfit3(vector<int> &prices) {
    if(prices.empty()) return 0;
    int sSize = prices.size();
    //vector<vector<vector<int>>> dp(sSize,vector<vector<int>>(2,vector<int>(2,0)));
    int dp_1_0 = 0;
    int dp_1_1 = -prices[0];
    int dp_2_0 = 0;
    int dp_2_1 = INT_MIN;
    for(int i = 1;i<sSize;++i){
        dp_2_0 = max(dp_2_0,dp_2_1 + prices[i]);
        dp_2_1 = max(dp_2_1,dp_1_0 - prices[i]);
        dp_1_0 = max(dp_1_0,dp_1_1 + prices[i]);
        dp_1_1 = max(dp_1_1, - prices[i]);
    }
    return max(dp_1_0,dp_2_0);
}

int Algorithm::maxProfit4(int k, vector<int> &prices) {
    if(prices.empty()) return 0;
    if(k>=prices.size()/2)
        return maxProfit3(prices);
    int sSize = prices.size();
    vector<vector<vector<int>>> dp(sSize,vector<vector<int>>(k + 1,vector<int>(2,0)));
    for(int i = 0;i<sSize;++i){
        for(int j = k;j>0;--j){
            if(i == 0){
                dp[0][j][0] = 0;
                dp[0][j][1] = -prices[0];
                continue;
            }
            dp[i][j][0] = max(dp[i-1][j][0],dp[i-1][j][1] + prices[i]);
            dp[i][j][1] = max(dp[i-1][j][1],dp[i-1][j-1][0] - prices[i]);
        }
    }
    return dp[sSize - 1][k][0];
}

vector<int> Algorithm::countSmaller(vector<int> &nums) {
    int sSize = nums.size();
    vector<int> res(sSize,0);
    vector<int> index(sSize,0);
    for(int i = 0;i<sSize;++i) index[i] = i;
    mergeCountSmaller(nums,index,res,0,sSize - 1);
    return res;
}

void Algorithm::mergeCountSmaller(vector<int> &nums,vector<int> &index,vector<int> &res,int l,int r){
    if(l>=r){
        return;
    }
    int mid = l + (r - l)/2;
    mergeCountSmaller(nums,index,res,l,mid);
    mergeCountSmaller(nums,index,res,mid + 1,r);
    vector<int> tmp(r - l + 1);
    int i = l,j = mid + 1;
    int pos = l;
    while (i <= mid && j <= r){
        if(nums[index[j]] < nums[index[i]]){
            res[index[i]] += r - j + 1;
            tmp[pos] = index[i];
            i++;
        }
        else{
            tmp[pos] = index[j];
            j++;
        }
        pos++;
    }
    while (i <= mid){
        tmp[pos++] = index[i++];
    }
    while (j <= r){
        tmp[pos++] = index[j++];
    }
    std::copy(tmp.begin(),tmp.end(),index.begin() + l);
}

int Algorithm::calculateMinimumHP(vector<vector<int>> &dungeon) {
    int row = dungeon.size(), col = dungeon[0].size();
    dungeon[row - 1][col - 1] = -dungeon[row - 1][col - 1]<0?0:-dungeon[row - 1][col - 1];
    for (int i = col-2; i >= 0; --i)
        dungeon[row - 1][i] = max(0, dungeon[row - 1][i + 1]) - dungeon[row - 1][i];
    for (int i = row-2; i >= 0; --i)
        dungeon[i][col - 1] = max(0, dungeon[i + 1][col - 1]) - dungeon[i][col - 1];
    for (int i = row - 2; i >= 0; --i) {
        for (int j = col-2; j >= 0; --j)
            dungeon[i][j] = max(0, min(dungeon[i][j + 1], dungeon[i + 1][j]))- dungeon[i][j];
    }
    return dungeon[0][0]<0?1:dungeon[0][0]+1;
}

int Algorithm::numIdenticalPairs(vector<int> &nums) {
    int sSize = nums.size();
    vector<vector<int>> tmp(101);
    for(int i = 0;i<sSize;++i){
        tmp[nums[i]].push_back(i);
    }
    int ans = 0;
    for(int i = 0;i<=100;++i){
        if(tmp[i].empty()) continue;
        int len = tmp[i].size();
        ans += len*(len - 1)/2;
    }
    return ans;
}

int Algorithm::numSub(string s) {
    if(s.empty()) return 0;
    vector<long long> dp(s.size() + 1);
    dp[0] = 0;
    int index = 0;
    for(int i = 1;i<=s.size();++i){
        if(s[i - 1] != '1'){
            dp[i] = dp[i -1];
            index = i;
        }
        else{
            int n = i - index;
            dp[i]  = dp[i - 1] + n;
        }
    }
    return dp[s.size()] % (1000000007);
}

double Algorithm::maxProbability(int n, vector<vector<int>> &edges, vector<double> &succProb, int start, int end) {
    /*vector<vector<double>> adjacency(n,vector<double>(n,0.0));
    //构建邻接图
    for(int i = 0;i<edges.size();++i){
        adjacency[edges[i][0]][edges[i][1]] = succProb[i];
        adjacency[edges[i][1]][edges[i][0]] = succProb[i];
    }
    int count = 1;
    vector<std::pair<double,bool>> dis(n);
    for(int i = 0;i<n;++i){
        dis[i].first = adjacency[start][i];
        dis[i].second = false;
    }
    dis[start].first = 1.0;
    dis[start].second = true;
    while (count != n){
        double iMax = 0.0;
        int tmp = -1;
        for(int i = 0;i<n;++i){
            if(!dis[i].second && dis[i].first > iMax){
                iMax = dis[i].first;
                tmp = i;
            }
        }
        dis[tmp].second = true;
        ++count;
        for(int i = 0;i<n;++i){
            if(!dis[i].second && dis[tmp].first * adjacency[tmp][i] > dis[i].first){
                dis[i].first = dis[tmp].first * adjacency[tmp][i];
            }
        }
    }
    return dis[end].first;*/

    vector<vector<pair<double,int>>> graph (n,vector<pair<double,int>>());
    for (int i = 0; i < edges.size(); ++i) {
        auto e = edges[i];
        graph[e[0]].push_back({succProb[i],e[1]});
        graph[e[1]].push_back({succProb[i],e[0]});
    }
    vector<int> visited(n,0);
    priority_queue<pair<double,int>> q;
    q.push({1,start});
    while(!q.empty()) {
        auto p = q.top();
        q.pop();
        auto curProb = p.first;
        auto curPos = p.second;
        if (visited[curPos]) continue;
        visited[curPos] = 1;
        if (curPos == end) return curProb;
        for ( auto next : graph[curPos]){
            double nextProb = next.first;
            int nextPos = next.second;
            if (visited[nextPos]) continue;
            q.push({curProb*nextProb,nextPos});
        }
    }
    return 0;
}

vector<int> Algorithm::intersect(vector<int> &nums1, vector<int> &nums2) {
    if(nums1.size() < nums2.size())
        return intersect(nums2,nums1);
    unordered_map<int,int> aMap;
    vector<int> ans;
    for(const auto& n : nums2) aMap[n]++;
    for(const auto& n : nums1){
        if(aMap.find(n) != aMap.end() && aMap[n] != 0){
            ans.push_back(n);
            aMap[n]--;
        }
    }
    return ans;
}

int Algorithm::minimumTotal(vector<vector<int>> &triangle) {
    if(triangle.empty()) return 0;
    int row = triangle.size();
    for(int i = row - 2;i>=0;--i){
        int col = triangle[i].size();
        for(int j = 0;j<col;++j){
            triangle[i][j] += min(triangle[i + 1][j],triangle[i + 1][j + 1]);
        }
    }
    return triangle[0][0];
}

int Algorithm::numTree(int n) {
    vector<int> dp(n + 1,0);
    dp[0] = 1;
    dp[1] = 1;
    for(int i = 2;i<=n;++i){
        for(int j = 1;j<i + 1;++j){
            dp[i] += dp[j - 1]*dp[i - j];
        }
    }
    return dp[n];
}

bool Algorithm::isBipartite(vector<vector<int>> &graph) {
    int unColor = 0;
    int red = 1;
    int green = 2;
    vector<int> isNodeColor(graph.size(),0);
    for(int i = 0;i<graph.size();++i){
        if(isNodeColor[i] == unColor){
            queue<int> q;
            q.push(i);
            isNodeColor[i] = red;
            while (!q.empty()){
                int node = q.front();
                q.pop();
                int color = isNodeColor[node] == red ? green : red;
                for(int j = 0;j<graph[node].size();++j){
                    if(isNodeColor[graph[node][j]] == unColor) {
                        isNodeColor[graph[node][j]] = color;
                        q.push(graph[node][j]);
                    }
                    else if(isNodeColor[j] != color)
                        return false;
                }
            }
        }
    }
    return true;
}

int Algorithm::searchInsert(vector<int> &nums, int target) {
    int l = 0;
    int r = nums.size() - 1;
    while (l<=r){
        int mid = l + (r - l)/2;
        if(nums[mid] == target)
            return mid;
        if(nums[mid] < target)
            l = mid + 1;
        else if(nums[mid] > target)
            r = mid - 1;
    }
    return l;
}

vector<TreeNode *> Algorithm::generateTrees(int n) {
    if(!n) return vector<TreeNode*>();
    return back_generateTrees(1,n);
}

vector<TreeNode *> Algorithm::back_generateTrees(int l, int r) {
    vector<TreeNode*> ans;
    if(l > r){
        ans.push_back(nullptr);
        return ans;
    }
    for(int i = l;i<=r;++i){
        vector<TreeNode*> leftVt = back_generateTrees(l,i - 1);
        vector<TreeNode*> rightVt = back_generateTrees(i + 1,r);
        for(auto left : leftVt){
            for(auto right : rightVt){
                TreeNode* node = new TreeNode(i);
                node->left = left;
                node->right = right;
                ans.push_back(node);
            }
        }
    }
    return ans;
}

int Algorithm::minArray(vector<int> &numbers) {
    int sSize = numbers.size();
    int l = 0,r = sSize - 1;
    while(l<r){
        int mid = l + (r - l)/2;
        if(numbers[r] > numbers[l]) return numbers[l];
        if(numbers[mid] >= numbers[l] && numbers[mid] > numbers[r]){
            l = mid + 1;
        }else if(numbers[mid] <= numbers[r] && numbers[mid] < numbers[l]){
            r = mid;
        }else{
            l++;
            r--;
        }
    }
    return numbers[l];
}

bool Algorithm::KMPString(const string &s, const string &t) {
    int sSize = s.size();
    int tSize = t.size();
    vector<int> next(tSize,0);
    for(int i = 1,k = 0;i<tSize;++i){
        while(k>0 && t[i] != t[k])
            k = next[k - 1];
        if(t[i] == t[k])
            k++;
        next[i] = k;
    }
    for(int i = 0,k = 0;i<sSize;++i){
        while(k>0 && t[k] != s[i])
            k = next[k - 1];
        if(t[k] == s[i])
            k++;
        if(k == tSize) return true;
    }
    return false;
}

ListNode *Algorithm::reverseList(ListNode *head) {
    /*if(head == nullptr || head->next == nullptr)
        return head;
    ListNode* tmp = reverseList(head->next);
    head->next->next = head;
    head->next = nullptr;
    return tmp;*/
    ListNode* pre = nullptr;
    ListNode* cur = head;
    while(cur){
        ListNode* tmp = cur->next;
        cur->next = pre;
        pre = cur;
        cur = tmp;
    }
    return pre;
}

vector<int> Algorithm::preorderTraversal(TreeNode *root) {
    /*m_intVt = vector<int>();
    back_Preorder(root);
    return m_intVt;*/
    vector<int> ans;
    stack<TreeNode*> s;
    s.push(root);
    while(!s.empty()){
        TreeNode* tmp = s.top();
        s.pop();
        ans.push_back(tmp->val);
        if(tmp->right) s.push(tmp->right);
        if(tmp->left) s.push(tmp->left);
    }
    return ans;
}

void Algorithm::back_Preorder(TreeNode *root) {
    if(root == nullptr) return;
    m_intVt.push_back(root->val);
    back_Preorder(root->left);
    back_Preorder(root->right);
}

vector<int> Algorithm::inorderTraversal(TreeNode *root) {
    if(root == nullptr)
        return {};
    stack<TreeNode*> s;
    while(!s.empty()||root){
        if(root){
            s.push(root);
            root = root->left;
        }else{
            root = s.top();
            s.pop();
            m_intVt.push_back(root->val);
            root = root->right;
        }
    }
    return m_intVt;
}

int Algorithm::minPathSum(vector<vector<int>> &grid) {
    if(grid.empty()) return 0;
    int m = grid.size();
    int n = grid[0].size();
    for(int i = 1;i<n;++i){
        grid[0][i] += grid[0][i - 1];
    }
    for(int i = 1;i<m;++i){
        grid[i][0] += grid[i - 1][0];
    }
    for(int i = 1;i<m;++i){
        for(int j = 1;j<n;++j){
            grid[i][j] += min(grid[i - 1][j],grid[i][j - 1]);
        }
    }
    return grid[m - 1][n - 1];
}

bool Algorithm::divisorGame(int N) {
    return N % 2 ? false : true;
}

bool Algorithm::isSubsequence(string s, string t) {
    int k = 0;
    for(int i = 0;i<t.size();++i){
        if(k == s.size()) return true;
        if(s[k] == t[i]){
            ++k;
        }
    }
    if(k == s.size()) return true;
    return false;
}

int Algorithm::maxDepth(TreeNode *root) {
    if(!root) return 0;
    int ans = 0;
    queue<TreeNode*> q;
    q.push(root);
    while(!q.empty()){
        int n = q.size();
        ++ans;
        for(int i = 0;i<n;++i){
            TreeNode* node = q.front();
            q.pop();
            if(node->left) q.push(node->left);
            if(node->right) q.push(node->right);
        }
    }
    return ans;
    //递归
    //return root ? max(maxDepth(root->left),maxDepth(root->right)) + 1 : 0;
}

int Algorithm::integerBreak(int n) {
    /*if(n<3) return 1;
    if(n == 3) return 2;
    int Three = n/3;
    int ans = 1;
    for(int i = 0;i<Three;++i)
        ans *= 3;
    int rest = n%3;
    if(rest == 1){
        ans /= 3;
        ans *= 4;
    }else if(rest == 2)
        ans *= 2;
    return ans;*/
    vector<int> dp(n + 1,0);
    for(int i = 2;i<=n;++i){
        for(int j = 1;j<i;++j)
            dp[i] = max({j * (i - j),j * dp[i - j],dp[j]});
    }
    return dp[n];
}

int Algorithm::findMagicIndex(vector<int> &nums) {
    /*for(int i = 0;i<nums.size();++i){
        if(nums[i] == i) return i;
    }
    return -1;*/
    return back_findMagicIndex(nums,0,nums.size() - 1);
}

int Algorithm::back_findMagicIndex(vector<int> &nums, int l, int r) {
    if(l>r)
        return -1;
    int mid = l + (r - l)/2;
    int left = back_findMagicIndex(nums,l,mid - 1);
    if(left > -1)
        return left;
    else if(nums[mid] == mid)
        return mid;
    return back_findMagicIndex(nums,mid + 1,r);
}






























