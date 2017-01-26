## tests:
board = initialize_board()
from random import random
w_pos = [random() for i in range(3)]
w_neg = [-random() for i in range(3)]
w_init = normalize_w([0] + w_pos + w_neg)
board[1][1] = "O"
board[1][0] = "X"
board[1][2] = "O"
board[2][2] = "X"
print("count_x(board):")
print(count_x(board))
print()

print("V_hat(w_init, board):")
print(V_hat(w_init, board))
print()

print("board:")
print_board(board)
print()

print("best_move(w_init, board):")
print_board(best_move(w_init, board))
print()

print("successor(w_init, board):")
print_board(successor(w_init, board)) 
print()

print("V_train(w_init, board):")
print(V_train(w_init, board))
print()

## How to set the lms_eta and lms_x???
lms_eta = 0.1
lms_x = [5 * random()] + [5 * random() for i in range(3)] + [-5 * random() for i in range(3)]
print("lms_x:")
print(lms_x)
print()

print("w_update(w_init, lms_eta, lms_x, board):")
print(w_update(w_init, lms_eta, lms_x, board))
print()

print("w_init:")
print(w_init)
print()

## start training:
max_steps = 5000

w_learned = training_w(w_init, lms_eta, lms_x)

print("w_learned:")
print(w_learned)
print()

print("w_learned[i] / w_init[i] for i = 1,...,6:")
for i in range(1,7):
    print(w_learned[i] / w_init[i])

# as one instance, let max_steps = 5000:
#     w_learned[i] / w_init[i] for i = 1,...,6:
#
#     0.5379678116335297
#     0.11204196139530746
#     1.4878147692229542
#     0.401145739080291
#     1.0952824182089669
#     0.3049206508828393
#
# So we find that w[3] does increase, as we expect; while w[6] does not so.
# However, let max_steps = 10000:
# >>> w_learned[i] / w_init[i] for i = 1,...,6:
#     0.8749184017319621
#     0.9230390901459481
#     1.7200297007491852
#     0.9860361859074219
#     2.129321681635566
#     1.2656473150321272
#
# and let max_steps = 50000:
#     w_learned[i] / w_init[i] for i = 1,...,6:
#     1.3901419195436366
#     0.7172834676898521
#     0.8807 476116869741
#     0.6176561742776161
#     1.0498407294397898
#     4.338208224456971
