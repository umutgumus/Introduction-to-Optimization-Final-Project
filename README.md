# Introduction-to-Optimization-Final-Project
Optimizing EV Charging for a Smarter Grid;
This project demonstrates how Linear Programming (LP) can be used to optimize electric vehicle (EV) charging schedules in order to reduce electricity costs and improve grid efficiency.
📌 Project Overview
In this simulation, a set of electric vehicles (EVs) arrive and depart at different times with varying energy demands. Electricity prices vary across 24 time slots. The objective is to determine the optimal charging plan that:
Fulfills each EV’s energy requirement within its available charging window.
Minimizes the total electricity cost.
Does not exceed the station's total power limit at any hour.
📈 Optimization Approach
We use:
PuLP: Python library for linear programming.
Synthetic EV data: Randomized arrival/departure times and energy demands.
Random hourly electricity prices between 0.15 and 0.30 $/kWh.
The LP model minimizes cost subject to constraints on demand, availability, and station capacity.
📊 Results
The optimized charging strategy significantly reduces total cost compared to a naive (uniform) charging method.
Cost savings and charging profiles are visualized using matplotlib.
