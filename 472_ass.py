import numpy as np
import math as m
import matplotlib.pyplot as plt
import random as rand
import scipy as sci

class value_iteration:

    #class constructor
    def __init__(self,states,actions,eta,beta):

        #initialize variables
        self.states = states
        self.actions = actions
        self.eta = eta
        self.beta = beta   

    #conditional probablity function
    def prob(self,to_state,from_state,from_action):
       
        if from_state == 'G' and from_action == 1:
            if to_state == 'G':
                p = 0.1
            if to_state == 'B':
                p = 0.9
        
        if from_state == 'G' and from_action == 0:
            if to_state == 'G':
                p = 0.9
            if to_state == 'B':
                p = 0.1
        
        if from_state == 'B' and from_action == 1:
            if to_state == 'G':
                p = 0.5
            if to_state == 'B':
                p = 0.5
        
        if from_state == 'B' and from_action == 0:
            if to_state == 'G':
                p = 0.9
            if to_state == 'B':
                p = 0.1

        return p

    #cost function implementation
    def calc_cost(self,state,action):
        
        if state == 'G' and action == 1:
            cost = eta -1
        elif state == 'B' and action == 1:
            cost = eta
        else:
            cost = 0

        return cost

     #plotting function
    
    #plotting function
    def plot(self,graph_data):
        fig, axes = plt.subplots(1,2)

        axes[0].scatter(x=np.linspace(0,len(graph_data['G']),len(graph_data['G'])),y=graph_data['G'])
        axes[1].scatter(x=np.linspace(0,len(graph_data['B']),len(graph_data['B'])),y=graph_data['B'])
        axes[0].set_title('Good state')
        axes[1].set_title('Bad state')
        axes[0].set_xlabel('Iteration number')
        axes[0].set_ylabel('Value')
        axes[1].set_xlabel('Iteration number')
        axes[1].set_ylabel('Value')

    
        plt.show()
   
    #Algorithm implementation
    def run(self):

        #specify what tolerance is sufficient for convergence
        tolerance = 1e-300

        #initialize lists for plotting
        graph_data = {'G':[],'B':[]}

        #initialize optimal policy to 0
        optimal_policy = {'B':0,'G':0}

        #initalize V to 0
        V = {'B':0,'G':0} #value from [state]

        #counter to check etamber of iterations
        n=0
        
        #begin algorithm
        while(1): 

            #reset variables
            V_new = {'B':0,'G':0}
            V_diff = 0

            #append values to lists for plotting
            graph_data['G'].append(V['G'])
            graph_data['B'].append(V['B'])

            #iterate through states
            for state in states:

                #reset minimum value
                V_min = 0

                #iterate through actions
                for action in actions:

                    #get terminal cost
                    check_V = self.calc_cost(state,action) 

                    #get expected value of next state
                    for next_state in states: 
                        check_V += beta*self.prob(next_state,state,action)*V[next_state]

                    #check if new action is optimal
                    V_min = min(V_min,check_V)

                    #update state value and optimal polciy if optimal
                    if V[state] > check_V:
                        optimal_policy[state] = action
            
                #update new state value
                V_new[state] = V_min

                #update max difference
                V_diff = max(V_diff,abs(V[state] - V_new[state]))
                
            #update Value dictionary
            V = V_new

            #increment counter
            n+=1

            #check if converged and if so, break
            if V_diff <= tolerance:
                break

        print(f'optimal_policy for states: {optimal_policy}')
        print(f'etam iterations: {n}')
        print(f'Value of each state = {V}')

        self.plot(graph_data)

class policy_iteration:
    
    #class constructor
    def __init__(self,states,actions,eta,beta):
        self.states = states
        self.actions = actions
        self.eta = eta
        self.beta = beta
    
    #conditional probablity function
    def prob(self,to_state,from_state,from_action):
       
        if from_state == 'G' and from_action == 1:
            if to_state == 'G':
                p = 0.1
            if to_state == 'B':
                p = 0.9
        
        if from_state == 'G' and from_action == 0:
            if to_state == 'G':
                p = 0.9
            if to_state == 'B':
                p = 0.1
        
        if from_state == 'B' and from_action == 1:
            if to_state == 'G':
                p = 0.5
            if to_state == 'B':
                p = 0.5
        
        if from_state == 'B' and from_action == 0:
            if to_state == 'G':
                p = 0.9
            if to_state == 'B':
                p = 0.1

        return p

    #cost function implementation
    def calc_cost(self,state,action):
            
        if state == 'G' and action == 1:
            cost = eta -1
        elif state == 'B' and action == 1:
            cost = eta
        else:
            cost = 0

        return cost

    #had issues with built in inversion function so made my own (only for 2x2)
    def invert_matrix(self,matrix):

        det = np.linalg.det(matrix)
        inv_mat = np.zeros((2,2))
        inv_mat[0][0] = matrix[1][1]/det
        inv_mat[0][1] = -1*matrix[0][1]/det
        inv_mat[1][0] = -1*matrix[1][0]/det
        inv_mat[1][1] = matrix[0][0]/det

        return inv_mat

    #calculate probability matrix for given policy
    def calc_P_gamma(self, policy):

        return np.array([[self.prob('B','B',policy[0]), self.prob('G','B',policy[0])],[self.prob('B','G',policy[1]), self.prob('G','G',policy[1])]])

    #Algorithm implementation
    def run(self):

        #index 0 for bad state, index 1 for good state

        #create a list of all possible policies
        policies = []
        for i in range (0,2):
            for j in range (0,2):
                policies.append([i,j])
        
        #init results list
        results = []

        #main loop
        for i in range(0,4):
            
            #get policy
            policy = policies[i]

            #calculate cost vector
            C_gamma = np.array([[self.calc_cost('B',policy[0])],[self.calc_cost('G',policy[1])]])

            #calculate probability matrix
            P_gamma = self.calc_P_gamma(policy)

            #calculate value of policy
            W = self.invert_matrix(np.eye(2, dtype = int) - self.beta*P_gamma) @ C_gamma

            #save results
            results.append(W)

        #find minimum value
        min_val = results[0]
        for i in range(1,4):
            if results[i][0] < min_val[0] and results[i][1] < min_val[1]:
                min_val = results[i]
                min_policy = policies[i]

        print(f'State B: u = {min_policy[0]}, W = {min_val[0]}')
        print(f'State G: u = {min_policy[1]}, W = {min_val[1]}')

class Q_learning:

    #class constructor
    def __init__(self,states,actions,eta,beta):

        #variables initialization
        self.states = states
        self.actions = actions
        self.eta = eta
        self.beta = beta
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.Q_table = {'B':[0,0],'G':[0,0]}
        self.counter = {'B':[0,0],'G':[0,0]}
    
    #transition_kernel
    def prob(self,to_state,from_state,from_action):
       
        if from_state == 'G' and from_action == 1:
            if to_state == 'G':
                p = 0.1
            if to_state == 'B':
                p = 0.9
        
        if from_state == 'G' and from_action == 0:
            if to_state == 'G':
                p = 0.9
            if to_state == 'B':
                p = 0.1
        
        if from_state == 'B' and from_action == 1:
            if to_state == 'G':
                p = 0.5
            if to_state == 'B':
                p = 0.5
        
        if from_state == 'B' and from_action == 0:
            if to_state == 'G':
                p = 0.9
            if to_state == 'B':
                p = 0.1

        return p

    #cost function implementation
    def calc_cost(self,state,action):
            
        if state == 'G' and action == 1:
            cost = eta -1
        elif state == 'B' and action == 1:
            cost = eta
        else:
            cost = 0

        return cost

    #choose action based on epsilon greedy policy
    def choose_action_greedy(self,state):

        temp_arr = []

        #explore or exploit
        if rand.random() < self.epsilon:
            #explore
            action = rand.choice(self.actions)
        else:
            #exploit
            for action in actions:
                temp_arr.append(self.Q_table[state][action])
            if temp_arr[0] < temp_arr[1]:
                action = 0
            if temp_arr[0] >= temp_arr[1]:
                action = 1

        #decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action

    #choose random action    
    def choose_action(self):

        #get random action
        action = rand.choice(self.actions)

        return action

    #update state 
    def get_next_State(self,state,action):

        #used python random.choices to get a random state based on the transition probabilities
        if state == 'G' and action == 1:
            next_state = rand.choices(['G','B'],weights=[0.1,0.9])
        if state == 'G' and action == 0:
            next_state = rand.choices(['G','B'],weights=[0.9,0.1])
        if state == 'B' and action == 1:
            next_state = rand.choices(['G','B'],weights=[0.5,0.5])
        if state == 'B' and action == 0:
            next_state = rand.choices(['G','B'],weights=[0.9,0.1])

        return next_state

    #update Q table
    def Q_learn(self,state,action,next_state):

        #convert next_state from list to etamber
        next_state = next_state[0]

        #calculate cost
        cost = self.calc_cost(state,action)

        #get optimalo cost for next state
        min_Q = min(self.Q_table[next_state][0],self.Q_table[next_state][1])

        #update Q table
        self.Q_table[state][action] = (1-1/(1+self.counter[state][action])) * self.Q_table[state][action] + (1/(1+self.counter[state][action])) * (cost + self.beta * min_Q)
        
        #update counter for learning rate
        self.counter[state][action] += 1
    
    #plottin function
    def plot(self,graph_data):

        fig, axes = plt.subplots(1,2)

        axes[0].scatter(x=np.linspace(0,len(graph_data['G']),len(graph_data['G'])),y=graph_data['G'])
        axes[1].scatter(x=np.linspace(0,len(graph_data['B']),len(graph_data['B'])),y=graph_data['B'])
        axes[0].set_title('Good state')
        axes[1].set_title('Bad state')
        axes[0].set_xlabel('Iteration etamber')
        axes[0].set_ylabel('Value')
        axes[1].set_xlabel('Iteration etamber')
        axes[1].set_ylabel('Value')

        plt.show()

    #algorithm implementation
    def run(self, etam_iterations):

        #choose an initial state
        state = 'B'

        #initialize lists for plotting
        graph_data = {'G':[],'B':[]}

        #simulation loop
        for t in range(etam_iterations):
            
            #get action
            action = self.choose_action()
            
            #get next state based on action
            next_state = self.get_next_State(state,action)
            
            #update Q table
            self.Q_learn(state,action,next_state)
            
            #update optimal Q values for plotting
            for state in states:
                if self.Q_table[state][0] < self.Q_table[state][1]:
                    graph_data[state].append(0)
                else:
                    graph_data[state].append(1)

            #increment state (index 0 to convert list to etamber)
            state = next_state[0]

        print(f'Q table: {self.Q_table}')
        #plot results
        self.plot(graph_data)

class convex_analyitical_method:

    #class constructor
    def __init__(self,states,actions,eta,beta):
        self.states = states
        self.actions = actions
        self.eta = eta
        self.beta = beta

    #conditional probability function
    def prob(self,to_state,from_state,from_action):
       
        if from_state == 'G' and from_action == 1:
            if to_state == 'G':
                p = 0.1
            if to_state == 'B':
                p = 0.9
        
        if from_state == 'G' and from_action == 0:
            if to_state == 'G':
                p = 0.9
            if to_state == 'B':
                p = 0.1
        
        if from_state == 'B' and from_action == 1:
            if to_state == 'G':
                p = 0.5
            if to_state == 'B':
                p = 0.5
        
        if from_state == 'B' and from_action == 0:
            if to_state == 'G':
                p = 0.9
            if to_state == 'B':
                p = 0.1

        return p

    #cost function implementation
    def calc_cost(self,state,action):
            
        if state == 'G' and action == 1:
            cost = eta -1
        elif state == 'B' and action == 1:
            cost = eta
        else:
            cost = 0

        return cost

    def run(self):

        #set up linear program

        #rearrange the equation from 7.32 in the lecture notes into matrix form with A_eqX = b_eq
        A_eq = np.array([[self.prob('B','B',0)-1,self.prob('B','B',1)-1,self.prob('B','G',0),self.prob('B','G',1)],
                         [self.prob('G','B',0),self.prob('G','B',1),self.prob('G','G',0)-1,self.prob('G','G',1)-1],
                         [1,1,1,1]])

        b_eq = np.array([[0],[0],[1]])

        #cost function to be used as c^Tx in the linear program
        c = np.array([[self.calc_cost('B',0)],[self.calc_cost('B',1)],[self.calc_cost('G',0)],[self.calc_cost('G',1)]]).transpose()

        #default lower bound is 0 so no need to specify
        x = sci.optimize.linprog(c,A_eq=A_eq,b_eq=b_eq)

        print(x.x)

        g1 = {0:0,0:0}
        gamma = {'B':g1,'G':g1}

        gamma_0_B = x.x[0] / (x.x[0] + x.x[1])
        gamma_1_B = x.x[1] / (x.x[0] + x.x[1])
        gamma_0_G = x.x[2] / (x.x[2] + x.x[3])
        gamma_1_G = x.x[3] / (x.x[2] + x.x[3])

        print(f'gamma_0_B = {gamma_0_B}')
        print(f'gamma_1_B = {gamma_1_B}')
        print(f'gamma_0_G = {gamma_0_G}')
        print(f'gamma_1_G = {gamma_1_G}')

#global parameters for the algorithms
eta = 0.7
states = ['B','G']
actions = [0,1]
beta = 0.6

def main():

    #run algorithms
    
    value_iteration(states,actions,eta,beta).run()

    policy_iteration(states,actions,eta,beta).run()

    Q_learning(states,actions,eta,beta).run(500000)

    convex_analyitical_method(states,actions,eta,beta).run()
    
    
if __name__ == '__main__':
    main()