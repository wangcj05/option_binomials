import numpy as np
import math
import igraph
from igraph import *
from scipy.stats import norm
import matplotlib.pyplot as plt
#import networkx as nx
#plotly.tools.set_credentials_file(username='menno9123', api_key='mF7CzAdgD4AD36Mn8Fiz')
"""
INITIALIZE THE TREE
"""
n_steps = 20
n_vertices = int(n_steps*(n_steps+1)/2)

"""
USER INPUT AND OPTION ATTRIBUTES
"""

option_geos = ['American','European']
option_types = ['call','put']

select_geo = option_geos[int(input('Enter 0 for American option or 1 for European option: '))]
select_type = option_types[int(input('Enter 0 for call option or 1 for put option: '))]
select_div = input('Underlying stock is dividend stock? Answer with True or False: ')

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
price_max = float(input('Enter the max/min stock price at the maturity of the option: ')) 

maturity = maturity/12
d_time = (maturity)/n_steps
r_free = 0.02
r_interest = 0.10

print('delta time:',d_time,'maturity:',maturity)

"""
FINANCE FUNCTIONS
"""
""" This function is used if sigma is calculated using 
def calculate_sigma(max_stock_price, current_stock_price, time_interval):
    up_max = max_stock_price / current_stock_price
    sigma = np.log(up_max)/np.sqrt(time_interval)
    #print('Sigma = ',abs(sigma))
    return abs(sigma)

sigma = calculate_sigma(price_max, price_u, maturity)
"""
def next_price(last_price, up_potential): #sigma is the volatility (expected price movement in precentage)
    up_price = last_price*up_potential#sqrt(d_time) is the time-adustment factor to scale volatility
    down_price = last_price*(1/up_potential)
    return up_price, down_price 

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
def make_tree(sigma):
    
    G = Graph(n_vertices, directed=True)
    total = 0
    upside = math.exp(sigma*math.sqrt(d_time))
    downside = math.exp(-sigma*math.sqrt(d_time))
    p_up = ((math.exp(r_interest*d_time))-downside)/(upside-downside)
    p_down = 1-p_up
    
    G.vs[0]['price_u'] = price_u
    G.vs[0]['prob'] = 1
    G.vs[0]['up'], G.vs[0]['down'] = next_price(G.vs[0]['price_u'], upside)
    
    for m in range(1,n_steps+1,1):
        total +=m
        first = total-m
        last = total-1
        #print(total, first, last, m)
        if m>1:
            G.add_edges([(first-m+1,first)])
            G.vs[first]['price_u'] = G.vs[first-m+1]['down']
            G.vs[first]['prob'] = G.vs[first-m+1]['prob']*p_down
            G.vs[first]['up'],G.vs[first]['down'] = next_price(G.vs[first]['price_u'], upside)
            
            for n in range(first+1,last):
                G.add_edges([(n-m,n),(n-m+1,n)])
                G.vs[n]['price_u'] = G.vs[n-m+1]['down']
                G.vs[n]['prob'] = G.vs[n-m+1]['prob']*p_down + G.vs[n-m]['prob']*p_up
                G.vs[n]['up'],G.vs[n]['down'] = next_price(G.vs[n]['price_u'], upside)
                
            G.add_edges([(last-m,last)])
            G.vs[last]['price_u'] = G.vs[last-m]['up']
            G.vs[last]['prob'] = G.vs[last-m]['prob']*p_up
            G.vs[last]['up'],G.vs[last]['down'] = next_price(G.vs[last]['price_u'], upside)
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
            payoff = strike-G.vs[j]['price_u']
            if payoff > 0:
                G.vs[j]['value'] = payoff
            else:
                G.vs[j]['value'] = 0
            #print(G.vs[j]['value'])
    else:
        print('The option is neither a call or put, please enter the right type')
        
    #calculating option values
    count_option = n_vertices-n_steps-1
    for i in range(n_steps-1,0,-1):
        #print('level: ',i)
        for k in range(count_option,count_option-i,-1):
            #if American option, option can be exercised early and has the value for depending 
            #on whether the payoff  upon exercising or payoff on holding the eoption is higher
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
            #if European option, option cannot be exercised before maturity date
            if select_geo == 'European':
                G.vs[k]['value'] = option_value(r_interest,d_time,p_up, G.vs[k+i+1]['value'], G.vs[k+i]['value'])
            #print(G.vs[k]['value'])
        count_option -= i
    return G 



#print('Black-Scholes formula results: ',black_scholes(price_u, maturity, 0.4))
    
"""
Loop for iterating through values of sigma and putting them in an array. find_nearest then finds the optimal sigma for 
which the option value at node 0 is closest to the real option values (e.g. from finance.yahoo)
"""
#option_values = np.zeros((100,2))
sigma_values = np.zeros((100,3))
for s in range(1, 101,1):
    sig = s*0.01
    G = make_tree(sig)
    sigma_values[s-1][0] = sig
    sigma_values[s-1][1] = G.vs[0]['value']
    sigma_values[s-1][2] = black_scholes(price_u, maturity, sig)
    
index_s = find_nearest(sigma_values[:,1], select_real)
optimal_sigma = sigma_values[index_s][0]
optimal_price = sigma_values[index_s][1]

print('The optimal sigma is: ',optimal_sigma, ' at price closest to real value: ',optimal_price)
print('The Black-Scholes formula returnes an option value of: ', black_scholes(G.vs[0]['price_u'],maturity,optimal_sigma))

G = make_tree(sigma_values[index_s,0]) #sigma_values[index_s,0]
G.vs[0:]['value'] = np.around(G.vs[0:]['value'],1)

x_values = G.vs[n_vertices-n_steps:]['price_u']
y_values = G.vs[n_vertices-n_steps:]['prob']

plt.subplot(2,1,1)
plt.plot(x_values, y_values, 'o-')
plt.title('Probability distribution (Binomial tree)')
plt.ylabel('Probability')
plt.xlabel('Underlying stock price at maturity')
plt.subplot(2,1,2)
plt.plot(sigma_values[1:,0],sigma_values[1:,1],'b',sigma_values[1:,0],sigma_values[1:,2],'r')
plt.title('Black-Scholes compared to binomial calculations of ')
plt.ylabel('Option value at node 0')
plt.xlabel('Sigma')
plt.legend(['Binomial','Black-Scholes'])

plt.tight_layout()
plt.show()

plot(G,'output.pdf', layout = 'kk',root=0, vertex_label = G.vs['prob'],bbox = (2000,2000), vertex_shape = 'rectangle', vertex_size = 35)


