import numpy as np
import chess

#PyTorch library in https://github.com/pytorch/pytorch
import torch as tc

def num_moves(moves, gt):
  nxt_moves = moves
  
  if gt == 5:
    nxt_moves = nxt_moves + 1
  
  return nxt_moves

def piece_value(piece) -> int:
  val = 0
  
  if piece in ["p", "P"]:
    val = 1
    if piece == "p":
        val = -val

  elif piece in ["r", "R"]:
    val = 4
    if piece == "r":
        val = -val
  
  elif piece in ["b", "B"]:
    val = 3
    if piece == "b":
        val = -val

  elif piece in ["n", "N"]:
    val = 2
    if piece == "n":
        val = -val

  elif piece in ["q", "Q"]:
    val = 5
    if piece == "q":
        val = -val
    
  elif piece in ["k", "K"]:
    val = 6
    if piece == "k":
        val = -val
    
  
  return val

def piece_owner(piece):
  pl = 0

  if piece in ["p", "b", "n", "r", "q", "k"]:
    pl = 1

  return pl 

def score_nn(state):
    pl, num_mv, brd, moves, gt = state
    score = 0

    brd_c = brd.copy()
    
    sign = +1 if (pl == 0 and num_mv < moves) or (pl == 1 and num_mv == moves) else -1
    
    last_move = brd_c.move_stack[-1]
    
    brd_c.pop()
    
    if brd_c.is_checkmate() == True:
      for i in range(0,64):
        piece = str(brd.piece_at(i))
        piece_val = piece_value(piece)
        score = 10000          #score + piece_val
      if brd.turn == True:
        score = -score

    elif brd_c.is_stalemate() == True or brd_c.can_claim_draw() == True:
      for i in range(0,64):
        piece = str(brd.piece_at(i))
        piece_val = piece_value(piece)


        score = score + piece_val   
        
    else:
      for i in range(0,64):
        piece = str(brd.piece_at(i))
        piece_val = piece_value(piece)

        score = score + piece_val 
        
      if brd_c.is_capture(last_move) == True:
          score = score + sign*(3**num_mv)
          
      elif brd.is_check() == True:
          score = score + sign*200
            
    return score


def board_to_arr(state):
    brd = state[2].copy()
    
    board_arr = []
    
    
    for i in reversed(range(8)):
        row = []
        for j in range(8):
            piece = str(brd.piece_at(8*i + j))
            row.append(piece_value(piece))
        
        board_arr.append(row)
            
    
    
    return np.array(board_arr)

def create_sample(state):
    
    brd_arr = board_to_arr(state)
    target = score_nn(state)
    
    #brd_arr = tc.Tensor(brd_arr)
    #target = tc.tensor([target])
    
    
    return (brd_arr, target)



        


