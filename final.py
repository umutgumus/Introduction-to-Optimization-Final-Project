# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 15:44:29 2025

@author: faruk
"""

# %% importing libraries

import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# %% creating synthetic data

np.random.seed(35) # random seed to get same results

time_slots = list(range(24)) # 24 hours

price_per_hour = np.random.uniform(0.15, 0.30, size=24)  # random price per hour for simulation

num_evs = 5 # number of EV

# synthetic data of arrival/departure & demand
ev_data = []
for i in range(num_evs):
    arrival = np.random.randint(0, 18)  # random arrival time between 0 to 15
    departure = np.random.randint(arrival + 4, 24) # random departure time (at least 4 hour)
    demand = np.random.uniform(10, 40) # total energy requeirement
    ev_data.append({
        'arrival': arrival,
        'departure': departure,
        'demand': demand
    })

print("Example EV data:")
for i, ev in enumerate(ev_data):
    print(f"EV {i+1}: Arrival={ev['arrival']}, Departure={ev['departure']}, Demand={ev['demand']:.2f} kWh")


# %% building LP model

model = pulp.LpProblem("EV_Charging_Schedule", pulp.LpMinimize)

# defining decision variable 
E = pulp.LpVariable.dicts("E",
                          ((ev, t) for ev in range(num_evs) for t in time_slots),
                          lowBound=0)

# %% objective function

model += pulp.lpSum([price_per_hour[t] * E[(ev, t)] for ev in range(num_evs) for t in time_slots])

# %% constraints

# every ev should be fully charged
for ev in range(num_evs):
    model += pulp.lpSum([E[(ev, t)] for t in time_slots]) >= ev_data[ev]['demand']

# ev's can only be charged when they are available
for ev in range(num_evs):
    for t in time_slots:
        if t < ev_data[ev]['arrival'] or t >= ev_data[ev]['departure']:
            model += E[(ev, t)] == 0

# station capacity will not be exceeded
station_power_limit = 50  # kW
for t in time_slots:
    model += pulp.lpSum([E[(ev, t)] for ev in range(num_evs)]) <= station_power_limit


# %% solving

model.solve()
print("\nSolution Status:", pulp.LpStatus[model.status])


# %% results

# charging profile of every ev
result = pd.DataFrame(0, index=time_slots, columns=[f'EV_{ev+1}' for ev in range(num_evs)])

ev_costs = []

for ev in range(num_evs):
    cost = sum([price_per_hour[t] * E[(ev, t)].varValue for t in time_slots])
    ev_costs.append(cost)
    print(f"EV {ev+1} Cost: ${cost:.2f}")
    for t in time_slots:
        result.iloc[t, ev] = E[(ev, t)].varValue
        
total_cost = sum(ev_costs)
print(f"\nTotal Cost for All EVs: ${total_cost:.2f}")

print("\nHourly Charging Plan:")
print(result)

# total cost without optimization

naive_costs = []
naive_result = pd.DataFrame(0, index=time_slots, columns=[f'EV_{ev+1}' for ev in range(num_evs)])

for ev in range(num_evs):
    arrival = ev_data[ev]['arrival']
    departure = ev_data[ev]['departure']
    slots = list(range(arrival, departure))
    slots_len = len(slots)
    demand = ev_data[ev]['demand']
    equal_energy = demand / slots_len

    for t in slots:
        naive_result.iloc[t, ev] = equal_energy

    # Bu EV'nin maliyeti
    cost = sum([price_per_hour[t] * equal_energy for t in slots])
    naive_costs.append(cost)

# Her zaman slotunda toplam naive yük
naive_result['Total'] = naive_result.sum(axis=1)

# Naive total cost
naive_total_cost = sum(naive_costs)

print("\nNaive cost per EV (no optimization):")
for ev, cost in enumerate(naive_costs):
    print(f"EV {ev+1}: ${cost:.2f}")

print(f"\nNaive total cost for all EVs: ${naive_total_cost:.2f}")

percent_saving = ((naive_total_cost - total_cost) / naive_total_cost) * 100

print(f"\nPercentage Saving due to Optimization: {percent_saving:.2f}%")

# total charging profile
result['Total'] = result.sum(axis=1)


# %% visualization

# charging schedule
plt.figure(figsize=(12, 6))

for ev in result.columns[:-1]:
    plt.plot(result.index, result[ev], label=ev, linewidth=1.5)

plt.plot(result.index, result['Total'], label='Total Load', linewidth=3, color='black', linestyle='--')

plt.plot(result.index, price_per_hour * 50, '--', label='Price Indicator')
plt.xlabel('Time Slot (Hour)')
plt.ylabel('Energy Assigned (kWh)')
plt.title('EV Charging Schedule')
plt.legend()
plt.grid(True)
plt.show()


# cost calculationc
plt.figure(figsize=(6, 5))
bars = plt.bar(['Naive', 'Optimized'], [naive_total_cost, total_cost],
               color=['salmon', 'lightgreen'])

plt.ylabel('Total Cost ($)')
plt.title(f'Total Cost Comparison\nSaving: {percent_saving:.2f}%')
plt.grid(axis='y')


for bar, cost in zip(bars, [naive_total_cost, total_cost]):
    plt.text(bar.get_x() + bar.get_width()/2, cost + 0.5,
             f"${cost:.2f}", ha='center')

plt.show()
