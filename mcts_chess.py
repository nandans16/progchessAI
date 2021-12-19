import chess
import numpy as np
import random

#PyTorch library in https://github.com/pytorch/pytorch
import torch as tc


from NN_training import Net
import NN_helpers as nnh

#Load the trained neural network
net = Net()
net.load_state_dict(tc.load('nn_1_weights.pth'))

def num_moves(moves, gt):
  nxt_moves = moves
  
  if gt == 5:
    nxt_moves = nxt_moves + 1
  
  return nxt_moves

def piece_value(piece) -> int:
  val = 0
  
  if piece in ["p", "P"]:
    val = 1

  elif piece in ["r", "R"]:
    val = 2
  
  elif piece in ["b", "B"]:
    val = 2

  elif piece in ["n", "N"]:
    val = 2

  elif piece in ["q", "Q"]:
    val = 3
  
  return val

def piece_owner(piece):
  pl = 0

  if piece in ["p", "b", "n", "r", "q", "k"]:
    pl = 1

  return pl 

#The code hereon is almost entirely based on the class notes of CIS 667, Dept. of EECS, Syracuse University
def score(state):
    pl, num_mv, brd, moves, gt = state
    score = 0

    brd_c = brd.copy()
    
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

        if piece_owner(piece) == 1:
          piece_val = -piece_val

        score + piece_val        

    return score

def get_player(state):
    return state[0]

def children_of(state):
    moves = state[3]
    gt = state[4]
    
    #Copy the board
    brd = state[2].copy()  
    
    #Legal moves for the current state
    lm = list(brd.legal_moves)

    children = []

    for i in range(len(lm)):
        pl = state[0]
        num_mv = state[1]
        brd_c = brd.copy()
        child = ()  #To hold the child of current state

        brd_c.push(lm[i])   #Perform each legal-move

        brd_cc = brd_c.copy()

        #Turn shifts to opponent when it's a check, reset number of moves to the opponent's equivalent
        if brd_cc.is_check() == True or brd_cc.is_game_over() == True or num_mv == 1:
          child = (1-pl, num_moves(moves, gt), brd_c.copy(), num_moves(moves, gt), gt)

        #board.push() automatically swaps turns after each move; for more than 1-move per player manually change it to indicate correct player's turn
        else:
          brd_c.turn = not(brd_c.turn)
          child = (pl, num_mv-1, brd_c.copy(), moves, gt)  #Decrement 1-move from player
        
        children.append(child)


    return list(children)

def is_leaf(state):
    children = children_of(state)
    value = score(state)
    return len(children) == 0 or value != 0

#This is almost entirely based on the class notes of CIS 667, Dept. of EECS, Syracuse University
class Node:
    num_nodes = 0

    def __init__(self,state):
        Node.num_nodes = Node.num_nodes + 1
        self.state = state
        self.visit_count = 0
        self.score_total = 0
        self.score_estimate = 0
        self.child_list = None

    def children(self):
        if self.child_list == None:
            self.child_list = list(map(Node, children_of(self.state)))
        return self.child_list

    def N_values(self):
        return [c.visit_count for c in self.children()]

    def Q_values(self):
        children = self.children()
        sign = +1 if self.state[0] == 0 else -1
        Q = [sign * c.score_total / (c.visit_count+1) for c in children]
        # Q = [sign * c.score_total / max(c.visit_count, 1) for c in children]
        return Q
    
    def P_values(self):
        children = self.children()
        P = []
        
        for c in children:
            brd = []
            brd_c = []
            brd_c.append(nnh.board_to_arr(c.state))
            brd.append(brd_c)
            brd = np.array(brd)
            brd = tc.Tensor(brd)
            
            st = []
            st_c = []
            st_c.append([c.state[0], c.state[1], c.state[3], c.state[4]])
            st.append(st_c)
            st = np.array(st)
            st = tc.Tensor(st)
            
            p = net(brd, st)
            
            P.append(p.item())
        
        return P

def exploit(node):
    return node.children()[np.argmax(node.Q_values())]

def explore(node):
    return node.children()[np.argmax(node.N_values())] # TODO

def uct(node):
    # max_c Qc + sqrt(ln(Np) / Nc)
    Q = np.array(node.Q_values())
    N = np.array(node.N_values())
    
    U = Q + np.sqrt(np.log(node.visit_count + 1) / (N + 1))
    return node.children()[np.argmax(U)]

choose_child = uct        #Function name changed to UCT 

def uct_nn(node):
    Q = np.array(node.Q_values())
    N = np.array(node.N_values())
    P = np.array(node.P_values())
    
    U = Q + P*(np.sqrt(np.log(node.visit_count + 1) / (N + 1)))
    return node.children()[np.argmax(U)]


def rollout(node, ctr):
    
    result = 0
    ctr = ctr-1
    
    if is_leaf(node.state): 
      result = score(node.state)

    elif ctr > 0: 
      result = rollout(choose_child(node), ctr)
    
    node.visit_count += 1
    node.score_total += result
    node.score_estimate = node.score_total / node.visit_count
    return result

def rollout_nn(node, ctr):
    
    result = 0
    ctr = ctr-1
    
    if is_leaf(node.state): 
      result = score(node.state)

    elif ctr > 0: 
      result = rollout(uct_nn(node), ctr)
    
    node.visit_count += 1
    node.score_total += result
    node.score_estimate = node.score_total / node.visit_count
    return result

def MCTS(state, ctr, num_ro):

  node = Node(state)

  for i in range(num_ro):
    rollout(node, ctr)
  
  new_node = uct(node)
  new_state = new_node.state
  
  
  return new_state

def MCTS_NN(state, ctr, num_ro):

  node = Node(state)

  for i in range(num_ro):
    rollout_nn(node, ctr)
  
  new_node = uct_nn(node)
  new_state = new_node.state
  
  
  return new_state

def baseline_AI(state):
  
  new_state = ()
  
  pl = state[0]
  num_mv = state[1]  
  brd = state[2].copy()
  moves = state[3]
  gt = state[4]
  
  lm = list(brd.legal_moves)

  nxt_mv = random.choice(lm)

  brd.push(nxt_mv)

  brd_c = brd.copy()

  if brd_c.is_check() == True or brd_c.is_game_over() == True or num_mv == 1:
    new_state = (1-pl, num_moves(moves, gt), brd, num_moves(moves, gt), gt)

  else:
    brd.turn = not(brd.turn)
    new_state = (pl, num_mv-1, brd, moves, gt)  #Decrement 1-move from player

  return new_state
