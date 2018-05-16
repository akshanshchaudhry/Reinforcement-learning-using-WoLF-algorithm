# Reinforcement-learning-using-WoLF-algorithm

This a project in which reinforcement machine learning is applied. The problem statement of the project is defined below. I have use WoLF (win or learn fast) algorithm to achieve the desired results.

##############################################################################################################################

Problem Statement:-

Two robots are operating in a factory to bring metal bars to two different production halls. The metal
bars are dispensed in one place, only one bar can be picked up at a time, and each robot can only carry
one bar at a time. Once a metal bar is picked up, a new one will appear at the dispenser with probability
0.5 every time step (every action taken corresponds to one time step). Each robot has 3 action choices. It
can either try to pick up a metal bar, deliver it to the production hall, or wait. If it tries to pick up a metal
bar, it will succeed with probability 0.5 (due to imprecisions in its programming) if there is a metal bar
available and fail if there is none available. If it tries to deliver a metal bar to the production hall, it will
succeed with probability 1 if it is holding a metal bar and fail otherwise. If it decides to wait it will stay
in its place. If both robots try to pick up a metal bar at the same time, they will both fail. Each robot receives
a payoff of 4 if it successfully delivers a metal bar to the production hall and incurs a cost of 1 if it tries
to pick up a metal bar or if it tries to deliver one to the production hall (reflecting the energy it uses up).
The wait action does not incur a cost.

##############################################################################################################################

The above problem statement is a Stochastic Game. In the end both the robots become intelligent enough to carry out the task independently.

Output:- In the end you will get the Nash Equilibrium of all the states in the game. The above game has 8 states.

Please go through the attached report in order to understand the concept.


References: -
a. http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf
b. http://www.masfoundations.org/mas.pdf
