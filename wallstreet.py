from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Flatten, Activation, Input,concatenate
import os.path
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pandas as pd
import os
from copy import deepcopy
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
np.random.seed(0)
style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
HOLD = 0
BUY = 1
SELL = 2
MEM_SIZE = 50
REWARDS = {
    'illegal_op': -1.0, # penalty if agent tries to sell when he has no stock or buys when he has no money
    'hold': -0.01,     # micro mort we don't want the agent to just sit there and do nothing
}
GAMMA = 0.95 # discount factor
EPSILON = 0.7 # used to control the agent's level of autonomy. For the first few trades we want the agent to explore
DECAY = 0.9 # after 60 trades the agent will be making its own decisions
MAX_QTY_PER_AGENT = 5.0 # amount of stock each agent has
MAX_CREDIT = 4.0 # how much credit can an agent draw

class Agent:
    def __init__(self, se, stock, cash):
        self.stock_exchange = se
        self.cash = cash
        self.qty = stock 
        self.mem = []
        self.last_action = 0
        self.last_reward = .0
        self.last_state = se.get_history()
        self.last_cash_balance = .0
        self.total_spent = stock * se.get_current_price()
        self.brain = self.build_agent_brain()
        self.eps = EPSILON
        self.max_purchase_price = stock * se.get_current_price()

    def build_agent_brain(self):
        # the state of the world aka the market trades for the last hour
        state = Input(shape=(60,1),dtype='float32')
        # the amount of money left for the agent to spend + the price of his most expensive stock
        portfolio = Input(shape=(2,),dtype='float32')
        s = LSTM(units=32)(state)
        s = Dense(1, activation='linear')(s)
        x = concatenate([s, portfolio],axis=1)
        x = Dense(32, activation='linear')(x)
        action = Dense(3, name='action', activation='linear')(x)
        model = Model(inputs=[state, portfolio], outputs=[action])
        model.compile(optimizer='adam', loss='mse',metrics=['mse'])
        return model
    
    def observe(self, state, reward):
        self.mem.append((self.last_state, self.last_action, 
                               self.last_cash_balance, self.last_reward, 
                               self.max_purchase_price))
        action = self.action(state)
        if action == SELL:
            self.stock_exchange.sell(self)
        elif action == BUY:
            self.stock_exchange.buy(self)
        else:
            self.stock_exchange.nop(self)
        if len(self.mem)>=MEM_SIZE:
            self.learn()
        self.last_action = action
        self.last_reward = reward
        self.last_cash_balance = self.cash
        self.last_state = deepcopy(state)

    def learn(self):
        for s,la,lcb,r,mp in self.mem:
            s =np.array(s).reshape(1,-1, 1)
            p = np.array([mp,lcb]).reshape(1, 2)
            a = self.brain.predict([s,p])
            target = r + GAMMA * np.max(a)
            a[0][la] = target
            self.brain.fit([s,p], a, epochs=1, verbose=0)
        self.mem = []

    def action(self, state):
        self.eps *= DECAY
        if np.random.rand() < self.eps:
            a = np.random.randint(0, 2)
        else:
            s = np.array(state).reshape(1,-1,1)
            p = np.array([self.max_purchase_price, self.cash]).reshape(1,2)
            a = self.brain.predict([s,p])
            a = np.argmax(a)
        return a
    
    def transfer(self, price):
        self.cash += price
        self.qty += -np.sign(price)
        earnings = price - self.max_purchase_price
        if self.qty == 0:
            self.max_purchase_price = 0.0
        elif np.sign(price) == -1: # agent has bought more stock
            # get the highest price that the agent has bought at since it'll dominate his expenditures and thus determine his earnings
            self.max_purchase_price = max(self.max_purchase_price, -price)
            return 0.0001
        return earnings


class StockExchange:
    def __init__(self, history):
        self.history = history
        self.buy_offers = []
        self.sell_offers = []
        self.bad_offers = []
        self.no_offers = []

    def buy(self, agent):
        if agent.cash + MAX_CREDIT > self.get_current_price():
            self.buy_offers.append(agent)
        else:
            self.bad_offers.append(agent)
    
    def nop(self,agent):
        self.no_offers.append(agent)
    
    def sell(self, agent):
        if agent.qty>0:
            self.sell_offers.append(agent)
        else:
            self.bad_offers.append(agent)

    def clear(self):
        cp = self.get_current_price() # current price
        sellers = []
        buyers = []
        bad_offers = self.bad_offers
        no_offers = self.no_offers
        transactions = 0
        # while there are eager buyers and sellers left keep making trades
        while len(self.buy_offers)>0 and len(self.sell_offers)>0:
            b = self.buy_offers.pop()
            s = self.sell_offers.pop()
            # take the buyers money and calculate their reward
            rb = b.transfer(-cp)
            # give the seller his due and calculate their reward
            rs = s.transfer(cp)
            buyers.append((b, rb))
            sellers.append((s, rs))
            transactions+=1
        # increase price by 2% if more people are buying else decrease it by 2% if more people are selling
        new_price = cp * 0.02 * np.sign(len(self.buy_offers)-len(self.sell_offers)) + cp 
        # use the line below if you want to change the price only if there are transactions
        # new_price = cp * 0.02 * np.sign(len(self.buy_offers)-len(self.sell_offers) if transactions>0 else 0) + cp
        self.history.append(new_price)
        no_offers.extend(self.buy_offers)
        no_offers.extend(self.sell_offers)
        self.buy_offers = []
        self.sell_offers = []
        self.bad_offers = []
        self.no_offers = []
        hist = self.get_history()
        # notify the sellers what happened and give them their rewards
        for s,rs in sellers:
            s.observe(hist, rs)
        # notify the buyers what happened
        for b,rb in buyers:
            b.observe(hist, rb)
        # tell the hodl gang that another eon has passed and they haven't done anything productive yet
        for n in no_offers:
            n.observe(hist, REWARDS['hold'])
        # punish the cheaters
        for i in bad_offers:
            i.observe(hist, REWARDS['illegal_op'])
    
    def get_history(self):
        return self.history[-60:]
    def get_current_price(self):
        return self.history[-1]

if __name__== '__main__':
    dt = pd.read_csv('Google_Stock_Price_Train.csv')[-60:]
    hist = list(dt['Open'].values/1000)
    stock_exchange = StockExchange(hist)
    agents = []
    for i in range(10):
        if i%2==0:
            a = Agent(stock_exchange, MAX_QTY_PER_AGENT, 0.0)
        else:
            a = Agent(stock_exchange, 0.0, 1.0)
        a.observe(hist, .0)
        agents.append(a)
    iters = 10
    while iters>0:
        stock_exchange.clear()
        iters-=1
    wealth = [a.cash for a in agents]
    print(wealth)
    print('total wealth: {}'.format(sum(wealth)))
    print('richest bot: {}'.format(max(wealth)))
    print('poorest bot: {}'.format(min(wealth)))
    graph_data = stock_exchange.history
    xs = []
    ys = []
    for x,y in enumerate(graph_data):
        xs.append(float(x))
        ys.append(float(y))
    ax1.clear()
    ax1.plot(xs, ys)
    plt.show()