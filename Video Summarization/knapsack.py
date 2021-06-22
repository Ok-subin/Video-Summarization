import numpy as np

'''
------------------------------------------------
Use dynamic programming (DP) to solve 0/1 knapsack problem
Time complexity: O(nW), where n is number of items and W is capacity

Author: Kaiyang Zhou
Website: https://kaiyangzhou.github.io/
------------------------------------------------
knapsack_dp(values,weights,n_items,capacity,return_all=False)

Input arguments:
  1. values: a list of numbers in either int or float, specifying the values of items
  2. weights: a list of int numbers specifying weights of items
  3. n_items: an int number indicating number of items
  4. capacity: an int number indicating the knapsack capacity
  5. return_all: whether return all info, defaulty is False (optional)

Return:
  1. picks: a list of numbers storing the positions of selected items
  2. max_val: maximum value (optional)
------------------------------------------------
'''
def knapsack_dp(values,weights,n_items,capacity,return_all=False):
    check_inputs(values,weights,n_items,capacity)

    table = np.zeros((n_items+1,capacity+1),dtype=np.float32)       # 행 개수 = segment 개수 , 열 개수 = 제한 프레임 수
    keep = np.zeros((n_items+1,capacity+1),dtype=np.float32)

    for i in range(1,n_items+1):        # 1 ~ segment 끝까지
        for w in range(0,capacity+1):       # 0 ~ 요약 영상에서 사용할 프레임 길이
            wi = weights[i-1] # weight of current item      # = segment별 프레임 수
            vi = values[i-1] # value of current item        # = segment별 score = seg_score
            if (wi <= w) and (vi + table[i-1,w-wi] > table[i-1,w]):
                # 1) 제한 프레임 길이보다 segment별 프레임 수가 적고, 2) (i번째 segment의 score + (i-1번째 (행) segment의 제한-현재 segment 길이번째 (열)의 값)이 (i-1번째 segment의 제한 길이번째 값)보다 크면
                # score의 합(묶음)이 가장 커지는 방향 vs score가 높은 순으로
                table[i,w] = vi + table[i-1,w-wi]
                keep[i,w] = 1
            else:
                table[i,w] = table[i-1,w]

    picks = []
    K = capacity

    for i in range(n_items,0,-1):
        if keep[i,K] == 1:
            picks.append(i)
            K -= weights[i-1]

    picks.sort()
    picks = [x-1 for x in picks] # change to 0-index


    if return_all:
        max_val = table[n_items,capacity]
        return picks,max_val
    return picks

def check_inputs(values,weights,n_items,capacity):
    # check variable type
    assert(isinstance(values,list))
    assert(isinstance(weights,list))
    assert(isinstance(n_items,int))
    assert(isinstance(capacity,int))
    # check value type
    assert(all(isinstance(val,int) or isinstance(val,float) for val in values))
    assert(all(isinstance(val,int) for val in weights))
    # check validity of value
    assert(all(val >= 0 for val in weights))
    assert(n_items > 0)
    assert(capacity > 0)

if __name__ == '__main__':
    values = [2,3,4]
    weights = [1,2,3]
    n_items = 3
    capacity = 3
    picks = knapsack_dp(values,weights,n_items,capacity)
    print (picks)
