# Topology Designer using Reinforcement Learning 
## Created by Nathan Brown during his PhD research within Clemson University's Mechanical Engineering Department

The series of codes found in this folder can be used to define the underlying RL environment and architecture used to allow a deep RL agent to design an optimal topology under various load cases.

Altering the bounded or loaded elements can be completed within the "config.txt" file. 
To run the code, download all components within this folder to your local machine and run the main.py file. 
Running the main.py file will run the topology design sequence using the deel RL agent who has been previously trained. The saved weights of the deep neural network used to defined the value function of the deep RL agent can be found in the "NN_Weights" folder.
Once the code has compelted running, the proposed topology design will be output to the "App_Data.txt" file as a boolean material representation for each element.
This output can be simply plotted to visualize the proposed topology design. 


An more interactive version of RL topology design can be found at https://github.com/garland3/rl_app_docker . 
The programs found within this folder should be used to experiment with the deep RL topology design environment, allowing a user to alter the underlying RL architecture if they so chose. 
