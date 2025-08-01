from collections import deque
import copy
import numpy as np

class Board:
    def __init__(self, rows, cols):
        self._last_black_move = None
        self._last_white_move = None
        self._blacks_prisoners = 0
        self._whites_prisoners = 0
        self._rows = rows
        self._cols = cols
        self.player = 1
        self._ko = None
        self._ko_player = None
        self._board = np.zeros((rows, cols))
        self._turn_board = np.zeros((rows, cols))
        self.turn = 1
        

    def get_board(self):
        return self._board


    def get_rows(self):
        return self._rows


    def get_cols(self):
        return self._cols


    def percent_filled(self):
        # print(np.count_nonzero(self._board))
        return np.count_nonzero(self._board) / (self._rows * self._cols)
    
    
    def _coord_outside_board(self, row, col):
        if row < 0 or row >= self._rows or col < 0 or col >= self._cols:
            return True
        return False
    
    
    def _get_connected(self, row, col, board=None):
        board = board if board is not None else self._board
        if self._coord_outside_board(row, col):
            raise ValueError("(", row, ",", col, ") is outside the bounds of the board; Cannot calculate liberties")
            
        chain = {(row, col)}
        neighbors = deque([(row+1, col), (row-1, col), (row, col+1), (row, col-1)])
        visited = {(row, col)}
        player = board[row][col]
        while len(neighbors) > 0:
            n_row, n_col = neighbors.popleft() # queue
            if self._coord_outside_board(n_row, n_col) or (n_row, n_col) in visited:
                continue
            visited.add((n_row, n_col))
            if board[n_row][n_col] == player:
                chain.add((n_row, n_col))
                new_neighbors = [(n_row+1, n_col), (n_row-1, n_col), (n_row, n_col+1), (n_row, n_col-1)]
                neighbors.extend(new_neighbors)
        return chain
    
    
    def _calculate_liberties(self, chain, board=None):
        board = board if board is not None else self._board
        visited, liberties = set(), set()
        for (row, col) in chain:
            visited.add((row, col))
            neighbors = deque([(row+1, col), (row-1, col), (row, col+1), (row, col-1)])
            for (n_row, n_col) in neighbors:
                if (n_row, n_col) in visited:
                    continue
                visited.add((n_row, n_col))
                if not self._coord_outside_board(n_row, n_col) and board[n_row][n_col] == 0:
                    liberties.add((n_row, n_col))
                
        return len(liberties)
           
        
    def get_legal_actions(self): #max 82 moves
        legal_actions = ["pass"]
        
        for row in range(self._rows):
            for col in range(self._cols):
                if (row, col) == self._ko and self._ko_player == -self.player:
                    continue
                elif self._board[row][col] != 0:
                    continue
                    
                board_copy = copy.deepcopy(self._board)
                board_copy[row][col] = self.player
                chain = self._get_connected(row, col, board=board_copy)
                if self._calculate_liberties(chain, board=board_copy) != 0:
                    legal_actions.append((row, col))
                    continue
                else:
                    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                        r, c = row + dr, col + dc
                        if 0 <= r < self._rows and 0 <= c < self._cols:
                            if board_copy[r][c] == -self.player:
                                opp_chain = self._get_connected(r, c, board=board_copy)
                                if self._calculate_liberties(opp_chain, board=board_copy) == 0:
                                    legal_actions.append((row, col))
                                    break
        # print(f"{len(legal_actions)} found for player: {self.player}")
        return legal_actions

    
    def is_game_over(self, board=None):
        board = board if board is not None else self._board
        if self._last_black_move == "pass" and self._last_white_move == "pass":
            return True
        for row in range(self._rows):
            for col in range(self._cols):
                if board[row][col] == 0:
                    return False
        return True
    
    
    def _calculate_territories(self, area_scoring, board=None): 
        # Note: the concept of stones being "dead" is ignored in this implementation
        board = board if board is not None else self._board
        black_territories, white_territories = 0, 0
        visited = set()
        for row in range(self._rows):
            for col in range(self._cols):
                if (row, col) in visited:
                    continue
                if board[row][col] == 0:
                    group = self._get_connected(row, col, board)
                    neighbors, contested = set(), False
                    for (r, c) in group:
                        for (dr, dc) in [(1,0), (-1,0), (0,1), (0,-1)]:
                            if 0 <= r + dr < self._rows and 0 <= c + dc < self._cols:
                                n = board[r+dr][c+dc]
                                if n == 0:
                                    continue
                                neighbors.add(n)
                                if len(neighbors) > 1:
                                    contested = True
                                    break
                        if contested:
                            break
                    if len(neighbors) == 1:
                        player = next(iter(neighbors))
                        if player == 1:
                            black_territories += len(group)
                        else:
                            white_territories += len(group)
                    visited.update(group)
                    
                elif area_scoring and (row, col) not in visited:
                    group = self._get_connected(row, col, board)
                    if board[row][col] == 1:
                        black_territories += len(group)
                    else:
                        white_territories += len(group)
                    visited.update(group)
        return {"Black": black_territories, "White": white_territories}
    
    
    def _get_newest_turn(self, chain, board=None, turn_board=None):
        board = board if board is not None else self._board
        turn_board = turn_board if turn_board is not None else self._turn_board
        newest = 1
        for (r, c) in chain:
            if self._turn_board[r][c] > newest:
                newest = self._turn_board[r][c]
        return newest
            
    
    def _capture_prisoners(self, board=None, turn_board=None):
        whites_prisoners, blacks_prisoners = 0,0
        board = board if board is not None else self._board
        turn_board = turn_board if turn_board is not None else self._turn_board
        visited, to_remove = set(), set()
        
        for row in range(self._rows):
            for col in range(self._cols):
                if (row, col) not in visited and board[row][col] != 0:
                    candidate_group = self._get_connected(row, col, board)
                    visited.update(candidate_group)

                    if self._calculate_liberties(candidate_group, board) == 0:
                        group_timestamp = self._get_newest_turn(candidate_group, board, turn_board)
                        this_player = board[row][col]
                        remove_this_group = False
                        visited_neighbors = set()
                        
                        for (r, c) in candidate_group:
                            for (dr, dc) in [(1,0), (-1,0), (0,1), (0,-1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < self._rows and 0 <= nc < self._cols:
                                    if (nr, nc) not in candidate_group and (nr, nc) not in visited_neighbors and board[nr][nc] == -this_player:
                                        opponent_group = self._get_connected(nr, nc, board)
                                        visited_neighbors.update(opponent_group)
                                        visited.update(opponent_group)
                                        
                                        if self._calculate_liberties(opponent_group, board) == 0: # ONE NEEDS TO GO
                                            neighbor_timestamp = self._get_newest_turn(opponent_group, board, turn_board)
                                            if neighbor_timestamp < group_timestamp:  # remove opponent_group
                                                to_remove.update(opponent_group)
                                            else:  # remove candidate_group and stop visiting neighbors
                                                to_remove.update(candidate_group)
                                                remove_this_group = True
                                            
                                            if remove_this_group:
                                                break   
                                        
                            if remove_this_group:
                                break
                        if len(to_remove) == 0:
                            to_remove.update(candidate_group)
                        
        # remove stones
        for (r, c) in to_remove:
            if board[r][c] == 1:
                whites_prisoners += 1
            else:
                blacks_prisoners += 1
                
            board[r][c] = 0
            turn_board[r][c] = 0

        return {"Black":blacks_prisoners, "White":whites_prisoners}
        
        
    def pass_turn(self):
        if self.player == 1:
            self._last_black_move = "pass"
        else:
            self._last_white_move = "pass"
        self.player *= -1
                    
                
    def move(self, row, col): # no simulations with move
        if row < 0 or row >= self._rows or col < 0 or col >= self._cols:
            raise ValueError("Invalid move: coordinate outside board")
        elif self._board[row][col] != 0:
            raise ValueError("Invalid move: coordinate not empty")
        elif (row, col) == self._ko and self._ko_player == -self.player:
            raise ValueError("Invalid move: Ko")
        newKo_r, newKo_c = None, None
        
        can_add = False
        
        # not immediately captured
        board_copy = copy.deepcopy(self._board)
        board_copy[row][col] = self.player
        chain = self._get_connected(row, col, board=board_copy)
        if self._calculate_liberties(chain, board=board_copy) != 0:
            can_add = True
            
        if not can_add: # captures an opponent piece
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                r, c = row + dr, col + dc
                if 0 <= r < self._rows and 0 <= c < self._cols:
                    if board_copy[r][c] == -self.player:
                        opp_chain = self._get_connected(r, c, board=board_copy)
                        if self._calculate_liberties(opp_chain, board=board_copy) == 0:
                            can_add = True
                            if len(opp_chain) == 1:
                                newKo_r = r
                                newKo_c = c
                            break

        if not can_add:
            raise ValueError("Invalid move: immediate capture")
                            
        self._board[row][col] = self.player
        self._turn_board[row][col] = self.turn
        self.turn+=1
        
        if self.player == 1:
            self._last_black_move = (row, col)
        else:
            self._last_white_move = (row, col)

        new_prisoners = self._capture_prisoners()
        blacks_prisoners, whites_prisoners = new_prisoners["Black"], new_prisoners["White"]
        self._blacks_prisoners += blacks_prisoners
        self._whites_prisoners += whites_prisoners
        
        
        if blacks_prisoners + whites_prisoners == 1:
            self._ko = (newKo_r, newKo_c)
            self._ko_player = self.player
        else:
            self._ko = None
            self._ko_player = None
        self.player *= -1

def board_to_tensor(board_instance):
    board = board_instance._board
    ko = board_instance._ko # (r,c)
    ko_player = board_instance._ko_player # 1 or -1
    player = board_instance.player # 1 or -1
    rows, cols = board.shape # 9,9
    last_black_move = board_instance._last_black_move # (r,c)
    last_white_move = board_instance._last_white_move # (r,c)
    turn_board = board_instance._turn_board
    turn = board_instance.turn # int
    tensor = np.zeros((10, rows, cols), dtype = np.float32)
    
    # Channels 0-2 are for board pieces
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == player:
                tensor[0][r][c] = 1  # current player's stones
            elif board[r][c] == -player:
                tensor[1][r][c] = 1  # opponent stones
            elif board[r][c] == 0:
                tensor[2][r][c] = 1  # empty positions
    # Channels 3-4 are for Ko     
    if ko:
        ko_r, ko_c = ko
        tensor[3][ko_r][ko_c] = 1  # ko position
        if ko_player == 1:
            tensor[4][:, :] = 1
        else:
            tensor[4][:, :] = -1
    
    # Channel 5 is for the player's turn it is next
    if player == 1:
        tensor[5][:, :] = 1  # 1 for black
    else:
        tensor[5][:, :] = 0  # 0 for white
    # Channel 6-7 is for last black/white moves
    tensor[6][:, :] = 0
    tensor[7][:, :] = 0
    if last_black_move:
        r, c = last_black_move
        tensor[6][r][c] = 1
    if last_white_move and last_white_move != "pass":
        r, c = last_white_move
        tensor[7][r][c] = 1
    # Channel 8 is for tracking when pieces were placed
    tensor[8] = turn_board
    
    # Channel 9 is for the current turn #
    tensor[9][0][0] = turn
        
    return tensor