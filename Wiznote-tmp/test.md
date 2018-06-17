# 链表

链表是一个动态的数据结构，不用事先预设链表的长度，当插入结点时，才会为该结点进行内存的分配，然后通过指针的指向来确保该结点被加入到链表当中。相比起数组，空间利用率更高，但是会造成内存的碎片化。

## 单向链表的节点定义

```c++
struct ListNode
{
	int val;
    ListNode* next;
    // ListNode(int x) : val(x), next(NULL) {}
    ListNode(int _val)
    {
    	val = _val;
        next = NULL;
    }
};
```

## 往链表末尾添加一个结点

```c++
void AddToTail(ListNode** head, int value)
{
	ListNode* p = new ListNode(value);
	if(head == nullptr)
    	*head = p;
  	else
    {
    	ListNode *tmp = *head;
        while(tmp->next != nullptr)
        	tmp = tmp->next;
       tmp->next = p;
    }
}
```

## 从尾到头打印链表

《剑指Offer》上的题目，题目链接: [牛客网-从尾到头打印链表](https://www.nowcoder.com/practice/d0267f7f55b3412ba93bd35cfa8e8035?tpId=13&tqId=11156&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

递归版本，有风险，可能当链表很长的时候会爆栈

```c++
void dfs(vector<int> &res, ListNode* node)
{
    if(node == nullptr)
        return ;
    dfs(res, node->next);
    res.push_back(node->val);
}
vector<int> printListFromTailToHead(ListNode* head) 
{
    vector<int> ans;
    dfs(ans, head);
    return ans;
}
```

非递归版本
```c++
vector<int> printListFromTailToHead(ListNode* head) {
    vector<int> ans;
    stack<int> s;
    while(head != nullptr)
    {
        s.push(head->val);
        head = head->next;
    }
    while(!s.empty())
    {
        ans.push_back(s.top());
        s.pop();
    }
    return ans;
}
```

## 反转链表

Leetcode题目链接：[反转链表](https://leetcode-cn.com/tag/linked-list/)

《剑指Offer》题目链接：[反转链表](https://www.nowcoder.com/practice/75e878df47f24fdc9dc3e400ec6058ca?tpId=13&tqId=11168&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

题目的要求是对整个链表的结构进行反转操作，主要是需要其实就是把每个结点指向前面的结点，然后头结点指向NULL，为了能让链表遍历下去，所以需要用个中间值tmp过度下，大致思路就是这样的。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* ans = NULL;
        ListNode* res = NULL;
        ListNode* p = head;
        while(p != nullptr)
        {
            if(p->next == nullptr)
                ans = p;
            ListNode* tmp = p->next;
            p->next = res;
            res = p;
            p = tmp;
        }
        return ans;
    }
};
```

## 链表中倒数第k个结点

输出该链表中倒数第k个结点

《剑指Offer》题目链接：[链表中倒数第k个结点](https://www.nowcoder.com/practice/529d3ae5a407492994ad2a246518148a?tpId=13&tqId=11167&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

大致思路就是：先让一个指针先走k步，然后再让第二指针重头开始，跟着第一个指针一起走，当第一个指针走到底时，第二个指针指向的结点就是要的答案。**有个坑点是：需要考虑k比链表长度大，即找不到第k个结点的情况**

```c++
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        int cnt = k;
        ListNode* p1 = pListHead;
        while(cnt > 0 && p1 != nullptr)
        {
            p1 = p1->next;
            cnt--;
        }
        if(cnt)
            return nullptr;
        ListNode* p2 = pListHead;
        while(p1 != nullptr)
        {
            p1 = p1->next;
            p2 = p2->next;
        }
        return p2;
    }
};
```

## 求两个链表的第一个公共结点

输入两个链表，找出它们的第一个公共结点。

《剑指Offer》题目链接：[求两个链表的第一个公共结点](https://www.nowcoder.com/practice/6ab1d9a29e88450685099d45c9e31e46?tpId=13&tqId=11189&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

假设两个链表存在公共节点，那么这两个链表的结尾一定是一样的，那么按照结尾对齐，然后把较长的一个链表先遍历掉那么长的部分，然后两个链表一起遍历，直到遇到相同的节点（返回结果）。否则就是没有相同的元素

```c++
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        int len1 = 0, len2 = 0;
        ListNode* p = pHead1;
        ListNode* q = pHead2;
        while(p!=NULL)
        {
            len1++;
            p = p->next;
        }
        while(q!=NULL)
        {
            len2++;
            q = q->next;
        }
        int k = 0;
        if(len1 > len2)
        {
            k = len1 - len2;
            while(k)
            {
                pHead1 = pHead1->next;
                k--;
            }
        }
        else if(len2 < len1)
        {
            k = len2 - len1;
            while(k)
            {
                pHead2 = pHead2->next;
                k--;
            }
        }
        while(pHead1 != nullptr)
        {
            if(pHead1 == pHead2)
                return pHead1;
            pHead1 = pHead1->next;
            pHead2 = pHead2->next;
        }
        return NULL;
    }
};
```

## 求链表中环的入口节点

一个链表中包含环，请找出该链表的环的入口结点。

《剑指Offer》题目链接：[求链表中环的入口节点](https://www.nowcoder.com/practice/253d2c59ec3e4bc68da16833f79a38e4?tpId=13&tqId=11208&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

采用快慢指针的方法

+ 首先用两个指针指向头结点(slow, fast)，slow一次走一步，fast一次走两步，直到slow==fast，当slow等于fast的时候就表示这个链表有环
+ 当确定链表有环的情况下，剩下来就是求环的入口节点位置
	+ 假设x表示头结点到环的节点数目，y表示从环入口节点到slow当前位置的节点个数，z表示slow距离环入口的节点个数
	+ 那么可以看出，slow经历了x+y个节点，fast经过了x+2y+z个结点，由于快慢指正的缘故，节点数有这样一种关系 x+2y+z = 2(x+y)，所以x=z
	+ 现在求环入口结点，只需要让fast指向头结点，然后和slow一起一次只移动一个结点，当相遇的那个结点就是环入口结点

![](https://github.com/CyC2018/Interview-Notebook/raw/master/pics/71363383-2d06-4c63-8b72-c01c2186707d.png)

```c++
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        if(pHead == nullptr)
            return nullptr;
        ListNode* fast = pHead;
        ListNode* slow = pHead;
        while(fast->next != nullptr)
        {
            fast = fast->next->next;
            slow = slow->next;
            if(fast == slow)
            {
                fast = pHead;
                while(slow != fast)
                {
                    slow = slow->next;
                    fast = fast->next;
                }
                return slow;
            }
        }
        return nullptr;
    }
};
```

## 复杂链表复制


输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

《剑指Offer》题目链接：[复杂链表复制](https://www.nowcoder.com/practice/f836b2c43afc4b35ad6adc41ec941dba?tpId=13&tqId=11178&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

整体思路是现将链表的每个结点后面都插入一个与自己相同的结点，然后在让插入的新结点的随机指向与初始的相同，然后再把复制的结点和原始的进行拆分，拆分出来的就是复制的链表。

```c++
/*
struct RandomListNode {
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) :
            label(x), next(NULL), random(NULL) {
    }
};
*/
class Solution {
public:
    RandomListNode* Clone(RandomListNode* pHead)
    {
        if(pHead == nullptr)
            return nullptr;
        RandomListNode* p = pHead;
        while(p != nullptr)
        {
            RandomListNode* tmp = new RandomListNode(p->label);
            tmp->next = p->next;
            p->next = tmp;
            p = tmp->next;
        }
        p = pHead;
        while(p != nullptr)
        {
            RandomListNode* tmp = p->next;
            if(p->random != nullptr)
                tmp->random = p->random->next;
            p = tmp->next;
        }
        p = pHead;
        RandomListNode* ans = p->next;
        while(p->next != nullptr)
        {
            RandomListNode* tmp = p->next;
            p->next = tmp->next;
            p = tmp;
        }
        return ans;
    }
};
```
