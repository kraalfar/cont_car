class Agent:
    def __init__(self):
        state_dim = 2
        action_dim = 15
        hd1 = 12
        hd2 = 12
        self.dqnA= torch.nn.Sequential(
            torch.nn.Linear(state_dim, hd1),
            torch.nn.ReLU(),
            torch.nn.Linear(hd1, hd2),
            torch.nn.ReLU(),
            torch.nn.Linear(hd2, action_dim))
        
        self.dqnB= torch.nn.Sequential(
            torch.nn.Linear(state_dim, hd1),
            torch.nn.ReLU(),
            torch.nn.Linear(hd1, hd2),
            torch.nn.ReLU(),
            torch.nn.Linear(hd2, action_dim))


        self.dqnA.load_state_dict(torch.load("agentCarA2.pkl"))
        self.dqnB.load_state_dict(torch.load("agentCarB2.pkl"))

    def get_probsA(self, state):
        ns = Variable(torch.Tensor(state))
        return self.dqnA(ns)
    
    def get_probsB(self, state):
        ns = Variable(torch.Tensor(state))
        return self.dqnB(ns)     

    def act(self, state, target=False):
        res = torch.argmax(self.get_probsA(state).data)
        return int(res)

