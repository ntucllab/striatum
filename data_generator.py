import striatum
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import linucb
import numpy as np
import matplotlib.pyplot as plt


def test1(alpha,T):
    ''' Linear payoff '''
    h1 = history.MemoryHistoryStorage()
    m1 = model.MemoryModelStorage()    
    d = 5
    actions = [1,2,3]
    policy = linucb.LinUCB(actions,h1,m1,alpha,d)
    context = np.random.uniform(0,1,(T,d))
    desired_action = np.zeros(shape = (T,1))
    actual_action = np.zeros(shape = (T,1))
    error = np.zeros(shape = (T,1))
    for t in range(T):
        if sum(context[t,:])<1:
            desired_action[t] = 1
        elif sum(context[t,:])<2.5:
            desired_action[t] = 2
        else:
            desired_action[t] = 3
        history_id, actual_action[t] = policy.get_action(context[t,:].tolist())

        if actual_action[t] != desired_action[t]:
            policy.reward(history_id,0)
            error[t] = 1
        else:
            policy.reward(history_id,1)
    return 1-sum(error)/T




def test2(alpha,T):
    # Nonlinear Payoff
    h1 = history.MemoryHistoryStorage()
    m1 = model.MemoryModelStorage()    
    d = 5
    actions = [1,2,3,4,5,6,7,8.9,10]
    policy = linucb.LinUCB(actions,h1,m1,alpha,d)
    context = np.random.uniform(0,2,(T,d))
    desired_action = np.zeros(shape = (T,1))
    actual_action = np.zeros(shape = (T,1))
    error = np.zeros(shape = (T,1))
    for t in range(T):
        if sum(context[t,:])<2.5:
            if context[t,1]<0.5:
                desired_action[t] = 1
            elif context[t,1]<1:
                desired_action[t] = 2
            elif context[t,1]<1.5:
                desired_action[t] = 3
            else:
                desired_action[t] = 4
        elif sum(context[t,:])<5:
            if context[t,2]**2<1:
                desired_action[t] = 4
            elif context[t,2]**2<2:
                desired_action[t] = 5
            elif context[t,2]**2<3:
                desired_action[t] = 6
            else:
                desired_action[t] = 7
        else:
            if np.sin(context[t,3])>0.5:
                desired_action[t] = 8
            elif np.sin(context[t,3])>0:
                desired_action[t] = 9
            else:
                desired_action[t] = 10
        
        history_id, actual_action[t] = policy.get_action(context[t,:].tolist())

        if actual_action[t] != desired_action[t]:
            policy.reward(history_id,0)
            error[t] = 1
        else:
            policy.reward(history_id,1)
            
    return 1-sum(error)/T

    
# Plot test1 result
test1_CTR = np.zeros(shape=(len(np.arange(0,3,0.1)),1))
i = 0
for alpha in np.arange(0,3,0.1):
    test1_CTR[i] = test1(alpha,10000)
    i = i + 1
plt.plot(test1_CTR)

# Plot test1 result
test1_CTR = np.zeros(shape=(len(np.arange(0,3,0.1)),1))
i = 0
for alpha in np.arange(0,3,0.1):
    test1_CTR[i] = test2(alpha,10000)
    i = i + 1
plt.plot(test1_CTR)

