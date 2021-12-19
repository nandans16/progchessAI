#Chess library for game environment, enforcing rules, making moves
import chess

import mcts_chess as mc
import time

#To convert state into the 8X8 chess board for the CNN and also to generate a score for the board at a given state
import NN_helpers as nn_help


#Used to store input and output data
#Found in the python core
import pickle


#Progrm to generate statistics to illustrate the performance of the AI
num_games = 20

game_type = 1

if __name__ == "__main__":
  
  ctr = 10       #Max depth of each rollout 
  num_ro = 20     #Number of rollout simulations per action/move
  t_limit = 120   #Draw if the game does not end in t_limit seconds

  nn_input = []
  nn_target = []

  for i in range(1, num_games+1):
    print("\n")
    print("Game "+str(i)+".......")
    board = chess.Board()
  
    if i % 4 == 0:
      game_type = game_type + 1
    
    moves = game_type

    if game_type == 5:
      moves = 1

    state = (0, moves, board.copy(), moves, game_type)
    
    start = time.time()
    
    t = time.time()

    while state[2].is_game_over() == False and (t-start) < t_limit:
      
      if state[0] == 0:
        state = mc.MCTS(state, ctr, num_ro)
        brd, targ = nn_help.create_sample(state)
        
        state_nn = (state[0], state[1], brd, state[3], state[4])
        
        nn_input.append(state_nn)
        nn_target.append(targ)

      
      elif state[0] == 1:
        state = mc.baseline_AI(state)
        brd, targ = nn_help.create_sample(state)
        
        state_nn = (state[0], state[1], brd, state[3], state[4])
        
        nn_input.append(state_nn)
        nn_target.append(targ)
      
      t = time.time()


  with open('train_data_input.pkl', 'ab') as f:
    pickle.dump(nn_input, f)

  with open('train_data_target.pkl', 'ab') as f:
    pickle.dump(nn_target, f)



