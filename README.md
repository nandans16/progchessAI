# progchessAI
This is a python-based AI that uses the Monte-Carlo Tree Search to play Standard, 2-move, 3-move, 4-move or progressive chess.  

*README*

This repository contains the python scripts that implement a chess-based AI and an interface to play five diffrent variants of chess. The variants include Standard or 1-move, 2-move, 3-move, 4-move and progressive chess. The following scripts are available in this repository:

-> chess_board_interface.py
-> MCTS_chess.py
-> chess_AI_performance.py


*EXISITING PACKAGES USED*:

The scripts use the python-chess library whose documentation can be found in: https://python-chess.readthedocs.io/en/latest/. The library can be installed with the command: pip install python-chess.

Other libraries that are used are numpy, random, time and matplotlib, which are commonly available python libraries. Random is used to select random elements from a list ad time is used to keep track of the time of the game. Matplotlib is used to generate the histograms in chess_AI_performance.py. 

*chess_board_interface.py*

This python script is the interface that allows the user to play chess. The chess board is displayed through a text-based visual indicating the pieces and their positions. There are five different types of games that can be played: Standard or 1-move, 2-move, 3-move, 4-move and progressive chess. The user can select the game type. The user can also choose which type of player they want to play as. Each player (player 0 and player 1) can be assigned to a human player, baseline AI player which makes random moves uniformly from all the possible moves available and an advanced AI that uses the Monte-Carlo Tree Search to play the game. 

When the player 0 plays as a baseline AI or an advanced AI player, the player can control when the move is made by pressing the ENTER key. 
For player 1, the player can control moves when they are playing as a human player.

*MCTS_chess.py*

This script contains the functions that are used to create nodes/states, evaluate the score at leaf nodes, generate children for each state, perform rollouts and the baseline AI that chooses random moves. 

Among these functions, the MCTS(state, ctr, num_ro): -> state, is responsible for performing multiple rollouts for each state where ctr is the parameter that controls the depth of rollout and num_ro is the number of rollouts. 

This script is based on the course notes in the CIS 667, Fall 2021 course: https://colab.research.google.com/drive/1JuNdI_zcT35MWSY4-h_2ZgH7IBe2TRYd?usp=sharing. Access is permitted only to students who are a part of the course. 

*chess_AI_performance.py*

This script is designed to simulate 100 games with the advanced AI competing against the baseline AI. The user can select the number of moves per turn or game type through an input. The second part of the code is used to generate the histogram plots for the number of nodes generated and the scores at the end of games. This requires the library matplotlib.   
     
