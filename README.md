# Traffic Control Intersection Task Reinforcement Learning Environment

This is an simulation of real life intersection for traffic control task on intersection in OpenAi Gym. The environment contains 13 lanes which consisted of 6 lanes for traffic influx, 5 lanes for traffic outpour and 2 connecting lanes.

There are 10 traffic lights throughout all intersection to control these intersections, each located according to the provided picture denoted by red or green lines.

The environment has been equipped with vehicle logic (other car detection and traffic light detection), and spawner which time is lane-based-randomized according to the seed passed.

![alt text](https://github.com/NyanNat/Intersection-Reinforcement-AIGym/blob/main/Traffic-light-environment-picture.png)

### Observation space
{{ Number of cars waiting on each intersection (10), Waiting index of each intersection (10), Time of execution (1) }}

### Action space
{{ Each Traffic Light states (10), Length of Action (3) }}, with a total of 3072 different possible actions for this environment

### Reward
Waiting index of a lane after it turned green * 0.001 + Number of cars reached its destination * 0.25
