## Homework Project 1 - Math is Fun

Solve 3 problems from the Euler Project using Python. Of the 3 problems, one must have been solved by fewer than 25,000 people, 1 fewer than 100,000 people and one fewer than 500,000 people. If you want to challenge yourself, choose all 3 problems from the category solved by fewer than 25,000 people. Write a function for each problem, and use nuympy-styple docstrings to annotate each function. Write a blog describing your solutions and how you approached each problem.


### Euler Project Problem 4 - Largest palindrome product 
_A palindromic number reads the same both ways. The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 × 99. 
Find the largest palindrome made from the product of two 3-digit numbers._

_Link to the problem: https://projecteuler.net/problem=4 (Solved by fewer than 500,000 people)_

Finding palindrome is definitely one of the most classical practice problems encountered by most newbie programmers. Other than just finding the palindromic number, this question takes it a littler further by asking you to find the largest palindrome made from the product of two 3-digit numbers. Since a palindromic pair is made of 2 numbers, the first approch that came to my mind is to use brute force - creating two loops that loop through all the numbers under a specified limit, which is also the integer n that the largest_palin function below takes. To answer this question specifically, this limit would be 1000 since looping thourgh 1 to 1000 will cover all 3-digit numbers.

As I loop through the specified range for the two lists made of i and j, I check to see if the product of two 3-digit numbers make a palindrome. Specifically, I convert the product of i and j to string and check to see if it equals the reverse ordered string of this product. If I find anything that satisfies this condition, I append the number to a list. Eventually, by calling max() on the list, I am able to find the largest palindrome made from the product of two 3-digit numbers. 

```

def largest_palin(n):
        
    """
    Takes in one integer n (number of digits), returns the largest palindrome made from the product of two n-digit numbers.
    """
    
    lst = []
    for i in range(0, n):  
        for j in range(0, n):  
            if str(i * j) == str(i*j)[::-1]: 
                lst.append(i*j) 
    return max(lst)


largest_palin(1000) #906609
```


### Euler Project Problem 34 - Digit factorials 

_145 is a curious number, as 1! + 4! + 5! = 1 + 24 + 120 = 145. Find the sum of all numbers which are equal to the sum of the factorial of their digits. Note: As 1! = 1 and 2! = 2 are not sums they are not included._

_Link to the problem: https://projecteuler.net/problem=34 (Solved by fewer than 100,000 people)_

If a number is equal to the sum of the factorial of their digits, then this number is called a curious number. To approach this problem, I start by importing the math package since I know at some point I would be calculating the factorial of some numbers and the `math.factorial` method would come in handy. Since the task here is related to fatorial of the digits of curious numbers, which can also have 9 options (factorial of 1 to 9), I precompute the factorials of those 9 numbers and save them in a dictionary structure so they can be easily refered later, and it won't be as computationally expensive either. 

To calculate factorials of all the digits of curious numbers, there are two steps. First, looping through all possible numbers from 3 on (since the prompt mentions that 1! = 1 and 2! = 2 are not sumes so they are not included), I convert each number in the loop to a string and compute the factorial based on this string value, which can be refered back to the factorial dictionary to find the corresponding factorial value, and sum up the factorials of the digits of this number. Next, I check if this number is curious by checking if its digit factorial sum is equal to the number itself. If this is the casse, I add the digits factorial sum to the total sum. Eventually, the total sum is what we are looking for, which is the sum of all curious numbers. 

In order to figure out the upper limit for the iterations, I did a quick Google search and found out that the upper limit for curious numbers is 50000. Therefore, I use 50000 as the upper limit/number of iterations and am able to solve the problem. 

```

import math

def find_fac_sum(n):
    
    """
    Takes in one integer n (upper limit of the iterations), returns the sum of all numbers which are equal to the sum of the factorial of their digits.
    """

    total_sum = 0
    factorials = {}
    
    for i in range(10):
        factorials[str(i)] = math.factorial(i) # precalculate factorials
    
    for num in range(3, n):
        fac_sum = 0
        for j in str(num):
            fac_sum += factorials[j]
        if fac_sum == num:
            total_sum += fac_sum
    return total_sum


find_fac_sum(50000) # 40730
```


### Euler Project Problem 145 - How many reversible numbers are there below one-billion? 

_Some positive integers n have the property that the sum [ n + reverse(n) ] consists entirely of odd (decimal) digits. For instance, 36 + 63 = 99 and 409 + 904 = 1313. We will call such numbers reversible; so 36, 63, 409, and 904 are reversible. Leading zeroes are not allowed in either n or reverse(n). There are 120 reversible numbers below one-thousand. How many reversible numbers are there below one-billion (10^9)?_
(solved by fewer than 25,000 people)

_Link to the problem: https://projecteuler.net/problem=145 (Solved by fewer than 25,000 people)_

To approch this problem, I start by forming a loop to loop through all numbers below the upper limit of iterations, which is one-billion here. Since the prompt specifies that leading zeros are not allowed in either the number itself or the reversed version of this number, I set a condition to exclude this situation before moving forward. Then, I compute the sum of the number itself and the reversed version of the number. After that, I create a variable `split`, which essentially splits the digits in this number and save them in a list. 

Next, for each number, I use the `all()` method to check if this number consists entirely of odd digits, and add one to my counter if this condition satisfies. Although this chuck of code takes a long time to run, since it loops through one billion numbers, computes the sum of those numbers and their reversed version, splits the digits of those numbers, and checks they are all odd digits, I am able to eventually find the correct answer. Future work would involve looking for a more efficient way to achieve the same result. 

```
def sum_reversible(n):
    
    """
    Takes in one integer n (upper limit of the iterations), returns the number of reversible numbers below the upper limit.
    """
    
    count = 0
    for i in range(1, n):
        if str(i)[-1] != '0':
            sum_ = i + int(str(i)[::-1])
            split = [int(b) for b in str(sum_)]
        
            if (all(split[idx] % 2 == 1 for idx in range(len(split)))): 
                count += 1
    return count

sum_reversible(1_000_000_000) # 608720
```

The code can also be found in this notebook. 
