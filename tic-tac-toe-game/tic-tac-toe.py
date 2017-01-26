# Exercise 1.5 of Machine Learning by T. Mitchell
# -----------------------------------------------
# All notations follow those in the texture.
# 
# T: play tic-tac-toe
# P: win the game
# E: play with ITSELF
# target function: V: board -> reals
# representation of target function:
#   V_hat(b) = w[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3] + w[4] * x[4] + w[5] * x[5] + w[6] * x[6]
#   where:
#     x[1]: numbers of one-in-one-line while the second and the third are still unfilled;
#     x[2]: numbers of two-in-one-line while the third is still unfilled;
#     x[3]: numbers of three-in-one-line.
#     x[4]: numbers of one-in-one-line while the second and the third are still unfilled for enermy;
#     x[5]: numbers of two-in-one-line while the third is still unfilled for enermy;
#     x[6]: numbers of three-in-one-line for enermy.
#   and by virtue of Python's terrible design, let x[0] = 1.
#  and V_hat(x[1] >=1) = 100, V_hat(x[4]>=1) = -100
# -------------------------------------------------
# Some problems are left:
# i)   how to set the input parameters, i.e. the initial w, the lms_x;
# ii)  how to set the `Successor`?
    
from math import sqrt

def initialize_board():
    """ null -> board
    """
    board = [0, 0, 0]
    for i in range(3):
        board[i] = ["?", "?", "?"]
    return board

def copy_board(board):
    result = initialize_board()
    for row in range(3):
        for column in range(3):
            result[row][column] = board[row][column]
    return result

def count_x(board):
    """ board -> x
    """
    x = [1, 0, 0, 0, 0, 0, 0]
    def count_in_line(lst):
        if lst.count("O") == 3:
            x[3] += 1
        elif lst.count("X") == 3:
            x[6] += 1
        elif lst.count("O") == 2 \
             and lst.count("?") == 1:
            x[2] += 1
        elif lst.count("X") == 2 \
             and lst.count("?") == 1:
            x[5] += 1
        elif lst.count("O") == 1 \
             and lst.count("?") == 2:
            x[1] += 1
        elif lst.count("X") == 1 \
             and lst.count("?") == 2:
            x[4] += 1
    # count in rows:
    for row in range(3):
        lst = [board[row][column] for column in range(3)]
        count_in_line(lst)
    # count in columns:
    for column in range(3):
        lst = [board[row][column] for row in range(3)]
        count_in_line(lst)
    # count in obliques:
    for obl in range(3):
        if obl == 0:
            lst = [board[0][0], board[1][1], board[2][2]]
            count_in_line(lst)
        elif obl == 2:
            lst = [board[2][0], board[1][1], board[0][2]]
            count_in_line(lst)
    return x

def V_hat(w, board):
    """ w * board -> Real
    """
    x = count_x(board)
    return sum(w[i] * x[i] for i in range(7))

def game_overQ(board):
    def flatten_board(board):
        return [y for x in board for y in x]
    result = False
    x = count_x(board)
    if x[3] >= 1 or x[6] >= 1: # someone wins
        True
    elif flatten_board(board).count("?") == 0: # none is empty
        True
    return result

def best_move(w, board):
    """ w * board -> board

        maximize V_hat
    """
    def arbitrary_move(board):
        """ board -> board
        """
        result = copy_board(board)
        for row in range(3):
            for column in range(3):
                if board[row][column] == "?": # empty?
                    result[row][column] = "O"
                    break
                break
            break
        return result
    if game_overQ(board) == True:
        print("Error: best_move: the input board has made the game over! No move is needed. an initialized board is as output.")
        return initialize_board()
    else:
        result = copy_board(board)
        for row in range(3):
            for column in range(3):
                next_try = copy_board(board)
                if board[row][column] == "?": # empty?
                    next_try[row][column] = "O"
                    if V_hat(w, next_try) > V_hat(w, result):
                        result = next_try
        if result == board:
            # that is, V_hat(w, board) has been
            # the maximum, thus no move has been
            # made yet. an arbitrary move is
            # called for
            result = arbitrary_move(board)
        return result

     
def successor(w, board):
    """ w * board -> board
    """
    def board_for_enermy(board):
        """ board -> board

            replace "O" <--> "X"
        """
        result = copy_board(board)
        for row in range(3):
            for column in range(3):
                if board[row][column] == "O":
                    result[row][column] = "X"
                elif board[row][column] == "X":
                    result[row][column] = "O"
        return result
    board = best_move(w, board)
    board = board_for_enermy(best_move(w, board_for_enermy(board)))
    return board

def V_train(w, board):
    """ w * board -> Real
    """
    return V_hat(w, successor(w, board))

def normalize_w(w):
    """ [Real] -> [Real]
    """
    norm = sqrt(sum(item**2 for item in w))
    return [item/norm for item in w]

def w_update(w, lms_eta, lms_x, board):
    """ w * lms_eta * lms_x * board -> w
    """
    return normalize_w([w[i] +\
                          lms_eta * lms_x[i] *\
                          (V_train(w, board) - V_hat(w, board)) \
                        for i in range(len(w))]

## default max_steps
max_steps = 5000

def training_w(w_init, lms_eta, lms_x):
    """ w_init * lms_eta * lms_x * board -> w
    """
    w = w_init
    board = initialize_board()
    for step in range(max_steps):
        if step >= max_steps:
            break
        else:
            x = count_x(board)
            if game_overQ(board) == True:
                board = initialize_board()
            else:
                board = successor(w, board)
                w = w_update(w, lms_eta, lms_x, board)
    return w

# Show the Result:
def print_board(board):
    for row in range(3):
        print(board[row])


# EOF
