# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.


## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.
## POLICY ITERATION ALGORITHM
Step 1: Start with a random policy and an arbitrary value function.

Step 2: Compute the value function for the current policy.

Step 3: Update the policy to be greedy with respect to the current value function.

Step 4: Repeat evaluation and improvement until the policy stabilizes.

Step 5: The final policy is optimal and provides the best actions for each state.

## POLICY IMPROVEMENT FUNCTION
### Name:DANIEL C
### Register Number:212223240023
```
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to implement policy improvement algorithm
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob, next_state,reward, done in P[s][a]:
          Q[s][a]+= prob*(reward+gamma*V[next_state]*(not done))
          new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi


pi_2 = policy_improvement(V1, P)
print("Name: Daniel C     ")
print("Register Number:    212223240023     ")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state)*100,
    mean_return(env, pi_2)))


V2 = policy_evaluation(pi_2, P)
print("Name:Daniel C      ")
print("Register Number:    212223240023     ")
print_state_value_function(V2, P, n_cols=4, prec=5)

# comparing the initial and the improved policy
if(np.sum(V1>=V2)==16):
  print("The Adversarial policy is the better policy")
elif(np.sum(V2>=V1)==16):
  print("The Improved policy is the better policy")
else:
  print("Both policies have their merits.")
```
## POLICY ITERATION FUNCTION
### Name:DANIEL C
### Register Number:212223240023
```
def policy_iteration(P, gamma=1.0, theta=1e-10):
   random_actions=np.random.choice(tuple(P[0].keys()),len(P))
   pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
   while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
   return V, pi

optimal_V, optimal_pi = policy_iteration(P)

print("Name:  Daniel C    ")
print("Register Number:   212223240023      ")
print('Optimal policy and state-value function (PI):')
print_policy(optimal_pi, P, action_symbols=('<','v','>','^'), n_cols=4)



print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, optimal_pi, goal_state=goal_state)*100,
    mean_return(env, optimal_pi)))

print("Name:Daniel C      ")
print("Register Number:  212223240023       ")
print_state_value_function(optimal_V, P, n_cols=4, prec=5)

```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
<img width="697" height="190" alt="image" src="https://github.com/user-attachments/assets/a73bd3f1-e5dc-41d8-93f1-f1a95f30d912" />


### 2. Policy, Value function and success rate for the Improved Policy
<img width="640" height="177" alt="image" src="https://github.com/user-attachments/assets/f1f94f4a-4cd3-4aec-9cfb-d1ec4350473c" />



### 3. Policy, Value function and success rate after policy iteration
<img width="771" height="58" alt="image" src="https://github.com/user-attachments/assets/013e04f1-cb94-457f-a4c3-998a6aa080a0" />

<img width="667" height="187" alt="image" src="https://github.com/user-attachments/assets/504c5149-90f2-4982-bfc1-b506369dff83" />



## RESULT:
Thus, the program to iterate Policy improvement and evaluation is implemented successfully

