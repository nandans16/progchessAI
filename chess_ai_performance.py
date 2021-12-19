import numpy as np
#Chess library in https://github.com/niklasf/python-chess
import chess
import mcts_chess as mc

#Matplotlib library in https://github.com/matplotlib/matplotlib
import matplotlib.pyplot as plt
import time

#PyTorch library in https://github.com/pytorch/pytorch
import torch

#Progrm to generate statistics to illustrate the performance of the AI
num_games = 100

if __name__ == "__main__":
  
  print("\n Game instances (program instances): \n")
  print("1. Standard Chess (1-move) \n")
  print("2. Chess (2-moves) \n")
  print("3. Chess (3-moves) \n")
  print("4. Chess (4-moves) \n")
  print("5. Progressive Chess (+1 move in every turn) \n")
  
  game_type = input("Enter the type of game that you would like to play: ")
  game_type = int(game_type)

#For a program instance size, use a histogram to plot the number of nodes generated over 100 games
#For a program instance size, use a histogram to plot the final game score

  num_nodes_list = []
  score_list = []
  avg_time_move_list = []

  ctr = 10       #Max depth of each rollout 
  num_ro = 20     #Number of rollout simulations per action/move
  t_limit = 60   #Draw if the game does not end in t_limit seconds

  for i in range(1, num_games+1):
    print("\n")
    print("Game "+str(i)+".......")
    board = chess.Board()
  
    moves = game_type

    if game_type == 5:
      moves = 1

    state = (0, moves, board.copy(), moves, game_type)
    
    mc.Node.num_nodes = 0

    start = time.time()
    t = time.time()

    move_time = []
    
    while state[2].is_game_over() == False and (t-start) < t_limit:
      
      
      if state[0] == 0:
        move_t = time.time()
        state = mc.MCTS(state, ctr, num_ro)
        end_move_t = time.time()

      
      elif state[0] == 1:
        state = mc.baseline_AI(state)

      t = time.time()
      move_time.append(end_move_t-move_t)

    
    
    num_nodes_list.append(mc.Node.num_nodes)
    score_list.append(mc.score(state))

    avg_time = sum(move_time)/len(move_time)
    avg_time_move_list.append(avg_time)

    print("\n", num_nodes_list)
    print("\n", score_list)
    print("\n", avg_time_move_list)

    game_str = str(game_type)

    if game_type == 5:
        game_str = "progressive"

  plt.hist(num_nodes_list, bins=20) 
  plt.xlabel("Number of nodes generated during each game")
  plt.ylabel("Count of games with respect to number of nodes generated")
  plt.title("Histogram of number of nodes generated every game")
  plt.savefig(r"Num_nodes_MCTS_chess_"+game_str+"_moves")
  plt.figure()

  plt.hist(score_list, bins=20) 
  plt.xlabel("Score at the end of each game")
  plt.ylabel("Count of games with respect to score")
  plt.title("Histogram of scores at the end of every game")
  plt.savefig(r"Score_MCTS_chess_"+game_str+"_move")
    
  plt.hist(score_list, bins=20) 
  plt.xlabel("Average time per move in each game")
  plt.ylabel("Count of games with respect to score")
  plt.title("Histogram of average times per move in every game")
  plt.savefig(r"Score_MCTS_chess_"+game_str+"_move")
