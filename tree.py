import numpy as np
from numpy import arange
import math

import igraph
from igraph import *

from scipy.stats import norm

#import networkx as nx
#plotly.tools.set_credentials_file(username='menno9123', api_key='mF7CzAdgD4AD36Mn8Fiz')
 
"""
INITIALIZE THE TREE
"""

n_steps = 20
n_vertices = int(n_steps*(n_steps+1)/2)
G = Graph(n_vertices, directed=True)
#G.add_edges([(0,1),(0,2),(1,3),(1,4),(2,4),(2,5)])
lay = G.layout('tree')

vertices = arange(0,n_vertices)

#Attributes of vertices option price, underlying price, delta hedge

"""
USER INPUT AND OPTION ATTRIBUTES
"""

option_geos = ['American','European']
option_types = ['call','put']

select_geo = option_geos[int(input('Enter 0 for American option or 1 for European option: '))]
select_type = option_types[int(input('Enter 0 for call option or 1 for put option: '))]
select_div = input('Underlying stock is dividen stock? Answer with True or False: ')

if select_div == 'True':
    q = float(input('What is the annual dividend yield?: '))
elif select_div == 'False':
    q = 0
else:
    print('Answer was neither True or False, zero dividend yield will be used: ')

select_real = float(input('Enter the real present value of the option: '))
maturity = float(input('Enter the maturity from present in whole months: '))
strike = float(input('Enter the strike price of the option: '))
price_u = float(input('Enter the price of the underlying asset: '))
price_max = float(input('Enter the max stock price at the maturity of the option: '))
#pay_off = #payoff at step N 
maturity = maturity/12
d_time = (maturity/12)/n_steps
r_free = 0.02
r_interest = 0.1
r_period = r_interest*(maturity/12)

"""
FINANCE FUNCTIONS
"""

def calculate_sigma(max_stock_price, current_stock_price, time_interval):
    up_max = max_stock_price / current_stock_price
    sigma = np.log(up_max)/np.sqrt(time_interval)
    #print('Sigma = ',abs(sigma))
    return abs(sigma)

sigma = calculate_sigma(price_max, price_u, maturity)

def next_price(sigma, d_time, last_price): #sigma is the volatility (expected price movement in precentage)
    up = last_price*math.exp(sigma*(math.sqrt(d_time)))#sqrt(d_time) is the time-adustment factor to scale volatility
    down = last_price*(1/upside)
    #print(up, down)
    #p_up = (r_interest-downside)/(upside-downside)
    #p_down = 1-p_up
    return up, down #p_up, p_down

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def option_value(r_interest, d_time, p, value_up, value_down):
    price_o = math.exp(-r_interest*d_time)*(p*value_up+(1-p)*value_down)
    #print(price_o)
    return price_o

def black_scholes(P, time_to_m, sig):
    PV_ex = strike*math.exp(-r_interest*time_to_m)
    d1 = math.log(P/PV_ex)/(sig*math.sqrt(time_to_m)) + (sig*math.sqrt(time_to_m))/2
    d2 = d1 - sig*math.sqrt(time_to_m)
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    price_bs = (nd1*P)-(nd2*PV_ex)
    return price_bs

"""
SETTING UP TREE
"""
#def make_tree(sigma):
total = 0
upside = math.exp(sigma*math.sqrt(d_time))
downside = math.exp(-sigma*math.sqrt(d_time))
#print('upside',upside,'downside',downside)
p_up = ((math.exp(r_interest*d_time))-downside)/(upside-downside)
p_down = 1-p_up
#print('p_up',p_up,'p_down',p_down)

G.vs[0]['price_u'] = price_u
G.vs[0]['up'], G.vs[0]['down'] = next_price(sigma, d_time, G.vs[0]['price_u'])

for m in range(1,n_steps+1,1):
    total +=m
    first = total-m
    last = total-1
    #print(total, first, last, m)
    if m>1:
        G.add_edges([(first-m+1,first)])
        G.vs[first]['price_u'] = G.vs[first-m+1]['down']
        G.vs[first]['up'],G.vs[first]['down'] = next_price(sigma, d_time, G.vs[first]['price_u'])
        
        for n in range(first+1,last):
            G.add_edges([(n-m,n),(n-m+1,n)])
            G.vs[n]['price_u'] = G.vs[n-m+1]['down']
            G.vs[n]['up'],G.vs[n]['down'] = next_price(sigma, d_time, G.vs[n]['price_u'])
            
        G.add_edges([(last-m,last)])
        G.vs[last]['price_u'] = G.vs[last-m]['up']
        G.vs[last]['up'],G.vs[last]['down'] = next_price(sigma, d_time, G.vs[last]['price_u'])
    #print('next level')
    
if select_type == 'call':
    for j in range (n_vertices-20,n_vertices,1):
        payoff = G.vs[j]['price_u']-strike
        if payoff > 0:
            G.vs[j]['value'] = payoff
        else:
            G.vs[j]['value'] = 0
        #print(G.vs[j]['value'])
elif select_type == 'put':
    for j in range(n_vertices-20,n_vertices,1):
        payoff = strike - G.vs[j]['price_u']
        if payoff > 0:
            G.vs[j]['value'] = payoff
        else:
            G.vs[j]['value'] = 0
        #print(G.vs[j]['value'])
else:
    print('The option is neither a call or put, please enter the right type')
    
count_option = n_vertices-n_steps-1
for i in range(n_steps-1,0,-1):
    #print('level: ',i)
    for k in range(count_option,count_option-i,-1):
        if select_geo == 'American':
            
            if select_type == 'call':
                payoff = G.vs[k]['price_u']-strike
                if payoff > option_value(r_interest,d_time,p_up, G.vs[k+i+1]['value'], G.vs[k+i]['value']):
                    G.vs[k]['value'] = payoff
                else:
                    G.vs[k]['value'] = option_value(r_interest,d_time,p_up, G.vs[k+i+1]['value'], G.vs[k+i]['value'])
                    
                    
            if select_type == 'put':
                payoff = payoff = strike - G.vs[j]['price_u']
                if payoff > option_value(r_interest,d_time,p_up, G.vs[k+i+1]['value'], G.vs[k+i]['value']):
                    G.vs[k]['value'] = payoff
                else:
                    G.vs[k]['value'] = option_value(r_interest,d_time,p_up, G.vs[k+i+1]['value'], G.vs[k+i]['value'])
            #print('vertex: ',k, G.vs[k+i+1]['value'], G.vs[k+i]['value'])
        if select_geo == 'European':
            G.vs[k]['value'] = option_value(r_interest,d_time,p_up, G.vs[k+i+1]['value'], G.vs[k+i]['value'])
        #print(G.vs[k]['value'])
    count_option -= i
#return G in case of 
#return G.vs[0]['value']

"""
#print('Black-Scholes formula results: ',black_scholes(price_u, maturity, 0.4))
sigma_values = np.zeros((100,2))
for s in range(1, 101,1):
    sigma = s*0.01
    sigma_values[s-1][0] = sigma
    sigma_values[s-1][1] = find_sigma(sigma)
    print('sigma: ',sigma_values[s-1][0],' value:',sigma_values[s-1][1])

index_s = find_nearest(sigma_values[:,1], select_real)
optimal_beta = sigma_values[index_s][0]
optimal_price = sigma_values[index_s][1]
print('The optimal beta is: ',optimal_beta, ' at price closest to real value: ',optimal_price)
"""
G = make_tree(0.35)
plot(G, layout = 'kk', label = G.vs['value'],sbox = [1200,1200], vertex_shape = 'rectangle', vertex_size = 15)


