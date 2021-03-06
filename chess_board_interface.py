#Chess library in https://github.com/niklasf/python-chess
import chess

import mcts_chess as mc

#Fucntion to make moves

def make_move(players, state):  #remove depth, moves, gt, players once it has been modified in play_chess
  pl_tm = state[0]
  num_mv = state[1]
  brd = state[2].copy()

  moves = state[3]

  gt = state[4]

  move = ""
  
  print("\n")
  print(brd)
  
  new_state = ()

  #If the current player is human
  if players[pl_tm] == "1":

    print("\nLegal moves for player "+str(pl_tm)+": \n")
    print(brd.legal_moves)
  
    move = input("\nPlease choose your move: ")
    move = brd.push_san(move)

    brd_c = brd.copy()

    if brd_c.is_check() == True or brd_c.is_game_over() == True or num_mv == 1:
      num_mv = mc.num_moves(moves, gt)
      pl_tm = 1-pl_tm
      moves = mc.num_moves(moves, gt)
      
    else:
      num_mv = num_mv-1
      brd.turn = not(brd.turn)

    new_state = (pl_tm, num_mv, brd.copy(), moves, gt)

  elif players[pl_tm] == "2":
    if pl_tm == 0:
      ent = input("\nPress Enter to request baseline AI to move: ")
      new_state = mc.baseline_AI((pl_tm,num_mv, brd, moves, gt))  #Set depth of MCTS
    
      
    if pl_tm == 1:
      new_state = mc.baseline_AI((pl_tm,num_mv, brd, moves, gt))  #Set depth of MCTS
      
  
  elif players[pl_tm] == "3":
    if pl_tm == 0:
      ent = input("\nPress Enter to request tree AI to move: ")
      new_state = mc.MCTS((pl_tm,num_mv, brd, moves, gt), 5, 10)  #Set depth of MCTS
      
    if pl_tm == 1:
      new_state = mc.MCTS((pl_tm,num_mv, brd, moves, gt), 5, 10)  #Set depth of MCTS

  elif players[pl_tm] == "4":
    if pl_tm == 0:
      ent = input("\nPress Enter to request NN + tree AI to move: ")
      new_state = mc.MCTS_NN((pl_tm,num_mv, brd, moves, gt), 5, 10)  #Set depth of MCTS
      
    if pl_tm == 1:
      new_state = mc.MCTS_NN((pl_tm,num_mv, brd, moves, gt), 5, 10)   
  return new_state  

#Plays chess after selecting players and game-type   
def play_chess(gt, pl_0, pl_1)-> tuple: 
  #Initializes a chess board 
  board = chess.Board()
  
  players = [pl_0, pl_1]
  pl_to_move = 0
  
  if gt != 5:
    num_moves = gt
  elif gt == 5:
    num_moves = 1

  moves = num_moves

  state = (pl_to_move, num_moves, board.copy(), moves, gt)

  brd = board.copy()

  while brd.is_game_over() == False:
    
    state = make_move(players, state)
    brd = state[2].copy()

  return state

#Main method      
if __name__ == "__main__":
  
  print("\n Game instances: \n")
  print("1. Standard Chess (1-move) \n")
  print("2. Chess (2-moves) \n")
  print("3. Chess (3-moves) \n")
  print("4. Chess (4-moves) \n")
  print("5. Progressive Chess (+1 move in every turn) \n")
  
  game_type = input("\nEnter the type of game that you would like to play:")
  game_type = int(game_type)

  print("\nChoose player 0: ")
  print("1. Human player: \n")
  print("2. Baseline AI player: \n")
  print("3. Advanced AI player: \n")
  print("4. Advanced AI with NN player: \n")

  valid = 0
  pl0 = 1
  
  while valid == 0:
    pl0 = input("Enter the type of type of player that you would like to play as:")
    
    if pl0 == '1' or pl0 == '2' or pl0 =='3' or pl0 =='4':
      valid = 1

  print("\nChoose player 1: ")
  print("1. Human player \n")
  print("2. Baseline AI player \n")
  print("3. Advanced AI player \n")
  print("4. Advanced AI with NN player: \n")

  valid = 0
  pl1 = 1
  
  while valid == 0:
    pl1 = input("\nEnter the type of type of player that you would like to play against:")
    
    if pl1 == '1' or pl1 == '2' or pl1 =='3' or pl1 == '4':
      valid = 1

  play_chess(game_type, pl0, pl1)

