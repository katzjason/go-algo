import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from flask import jsonify, request, Flask
import json



#### PASTING BOARD CLASS FROM GOOGLE COLAB
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
    #self._board = [[-1,-1,-1,-1,-1,-1,-1,-1,-1],
                  #  [-1,-1,-1,-1,-1, 0,-1,-1,-1],
                  #  [-1,-1,-1,-1,-1,-1,-1,-1,-1],
                  #  [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  #  [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  #  [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  #  [ 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  #  [ 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  #  [ 1, 1, 1, 1, 1, 1, 1, 0, 0]
                  # ]

  def get_board(self):
    return self._board

  def get_rows(self):
    return self._rows

  def get_cols(self):
    return self._cols
  
  def percentFilled(self):
    maxStones = self._rows * self._cols
    filledStones = 0
    for row in range(self._rows):
      for col in range(self._cols):
        if self._board[row][col] != 0:
          filledStones += 1
    return filledStones / maxStones

  def _calculate_liberties(self, row, col, visited):
    if row < 0 or row >= self._rows or col < 0 or col >= self._cols:
      raise ValueError("Coordinate outside board")
    elif self._board[row][col] == 0:
      raise ValueError("Empty coordinates can't have liberties")

    player = 1 if self._board[row][col] == 1 else -1

    if (col, row) not in visited:
      visited.add((col, row))

    liberties = 0
    surrounding = [(col-1, row), (col+1, row), (col, row+1), (col, row-1)]

    while len(surrounding) > 0:
      (surr_col, surr_row) = surrounding.pop()
      try:
        if (surr_col, surr_row) in visited:
          continue
        elif self._board[surr_row][surr_col] == 0:
          liberties += 1
        elif self._board[surr_row][surr_col] != player:
          visited.add((surr_col, surr_row))
        else:
          liberties += self._calculate_liberties(surr_row, surr_col, visited)
      except:
        continue
    return liberties


  def get_legal_actions(self):
    legal_actions = ["pass"] if self.percentFilled() > 0.75 else []
    i = len(legal_actions)
    for row in range(self._rows):
      for col in range(self._cols):
        added = False
        if self._board[row][col] != 0:
          continue
        elif (col, row) == self._ko and self._ko_player == -self.player: #ko
          continue

        self._board[row][col] = self.player # temporarily adding stone
        surrounding = [(col-1, row), (col+1, row), (col, row+1), (col, row-1)]
        while len(surrounding) > 0:
          (surr_col, surr_row) = surrounding.pop()
          try: # capture opponent?
            if self._board[surr_row][surr_col] == -self.player and self._calculate_liberties(row=surr_row, col=surr_col, visited=set()) == 0:
              self._ko = (surr_col, surr_row)
              self._ko_player = self.player
              legal_actions.append((col, row))
              i += 1
              added = True
              break
          except:
            continue

        if self._calculate_liberties(row=row, col=col, visited=set()) != 0 and not added: # not immediately captured
          legal_actions.append((col, row))
          self._ko = None
          self._ko_player = None
          i += 1

        self._board[row][col] = 0 # reset board
    #print("legal moves found: " + str(i) + " for player: " + str(self.player))
    return legal_actions


  def is_game_over(self):
    if self._last_black_move == "pass" and self._last_white_move == "pass" and self.player == 1:
      return True

    for row in range(self._rows):
      for col in range(self._cols):
        if self._board[row][col] == 0:
          return False

    return True


  def _calculate_territories(self, area_counting):
    territories_map = {}
    black_territories, white_territories = 0,0
    black_stones, white_stones = 0,0
    neutral = False

    coords = set()
    for row in range(self._rows):
      for col in range(self._cols):
        coords.add((col, row))

    # visited = set()
    while len(coords) > 0:
      for coord in coords:
        (x, y) = coord
        # visited.add((x, y))
        coords.remove((x, y))
        break

      if self._board[y][x] == 0:
        border_color = 0
        territory_size = 1
        to_visit = deque()
        to_visit.append((x+1, y))
        to_visit.append((x-1, y))
        to_visit.append((x, y+1))
        to_visit.append((x, y-1))

        while len(to_visit) > 0:
          (this_x, this_y) = to_visit.popleft()
          if this_x < 0 or this_x >= self._cols or this_y < 0 or this_y >= self._rows:
            continue

          if (this_x, this_y) not in coords: # visited
            if self._board[this_y][this_x] != 0:
              if border_color == 0:
                border_color = self._board[this_y][this_x]
              elif self._board[this_y][this_x] != border_color:
                neutral = True
            continue

          coords.remove((this_x, this_y))

          if self._board[this_y][this_x] == 0:
            territory_size += 1
            to_visit.append((this_x+1, this_y))
            to_visit.append((this_x-1, this_y))
            to_visit.append((this_x, this_y+1))
            to_visit.append((this_x, this_y-1))
          else:
            thisColor = self._board[this_y][this_x]
            if border_color == 0:
              border_color = thisColor
            elif thisColor != border_color:
              neutral = True

            if thisColor == 1:
              black_stones += 1
            else:
              white_stones += 1

        if not neutral:
          if border_color == 1:
            black_territories += territory_size
          elif border_color == -1:
            white_territories += territory_size

      elif self._board[y][x] == 1:
        black_stones += 1
      elif self._board[y][x] == -1:
        white_stones += 1

    if area_counting:
      territories_map["Black"] = black_territories + black_stones
      territories_map["White"] = white_territories + white_stones
    else:
      territories_map["Black"] = black_territories
      territories_map["White"] = white_territories

    return territories_map


  def get_score(self):
    # TODO prisoners?
    territories_map = self._calculate_territories(False)
    if self.player == 1:
      return territories_map["Black"] + self._blacks_prisoners
    else:
      return territories_map["White"] + self._whites_prisoners


  def count_group(self, col, row):
    if col < 0 or col >= self._cols or row < 0 or row >= self._rows:
      raise ValueError("Coordinate outside board")

    count = 1
    piece = self._board[row][col]
    visited = set()
    visited.add((col, row))
    surrounding = []
    surrounding.append((col-1, row))
    surrounding.append((col+1, row))
    surrounding.append((col, row-1))
    surrounding.append((col, row+1))

    while len(surrounding) > 0:
      (surr_col, surr_row) = surrounding.pop()
      if (surr_col, surr_row) in visited:
          continue
      else:
        visited.add((surr_col, surr_row))

      if surr_col < 0 or surr_col >= self._cols or surr_row < 0 or surr_row >= self._rows:
        continue
      elif self._board[surr_row][surr_col] == piece:
        count += 1
        surrounding.append((surr_col-1, surr_row))
        surrounding.append((surr_col+1, surr_row))
        surrounding.append((surr_col, surr_row-1))
        surrounding.append((surr_col, surr_row+1))

    return count

  def _flood_fill(self, col, row, player, visited):
    group = []
    if (col, row) not in visited:
      visited.add((col, row))
      if self._board[row][col] == player:
        group.append((col, row))
      else:
        return group
    surrounding = [(col+1, row), (col-1, row), (col, row+1), (col, row-1)]
    while len(surrounding) > 0:
      try:
        (surr_col, surr_row) = surrounding.pop()
        if (surr_col, surr_row) in visited:
          continue
        elif self._board[surr_row][surr_col] == 0:
          visited.add((surr_col, surr_row))
        elif self._board[surr_row][surr_col] != player:
          visited.add((surr_col, surr_row))
        else:
          group += self._flood_fill(surr_col, surr_row, player, visited)
      except:
        continue
    return group


  def _capture_prisoners(self):
    visited = set()
    for row in range(self._rows):
      for col in range(self._cols):
        if (col, row) not in visited:
          visited.add((col, row))
          if self._board[row][col] != 0:
            group = self._flood_fill(col, row, self._board[row][col], set())
            if self._calculate_liberties(row, col, set()) == 0:
              if self._board[row][col] == -1:
                self._blacks_prisoners += len(group)
              else:
                self._whites_prisoners += len(group)
              for (x, y) in group:
                self._board[y][x] = 0


  def move(self, col, row, passed):
    if row < 0 or row >= self._rows or col < 0 or col >= self._cols:
      raise ValueError("Coordinate outside board")
    elif self._board[row][col] != 0 and not passed:
      raise ValueError("Invalid move")

    if not passed:
      self._board[row][col] = self.player

    if self.player == 1:
      self._last_black_move = "pass" if passed else (col, row)
    else:
      self._last_white_move = "pass" if passed else (col, row)

    self._capture_prisoners()
    self.player *= -1

    return self

#### END BOARD CLASS FROM GOOGLE COLAB




def board_to_tensor(board, player, device="cpu"):
  # 3 Channels for the neural network
  #np_board = np.array(board)
  player_board = np.where(board == player, 1, 0)
  opponent_board = np.where(board == -player, 1, 0)
  empty_board = np.where(board == 0, 1, 0)
  tensor = np.stack([player_board, opponent_board, empty_board], axis=0)
  #return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(device)
  return torch.tensor(tensor, dtype=torch.float32)

def select_move(policy_net, board, device="cpu", temperature=1.0):
  state = board_to_tensor(board.get_board(), 1, device)
  #state = torch.tensor(board.get_board(), dtype=torch.float32).unsqueeze(0)
  log_probs = policy_net(state) / temperature
  move_probs = torch.exp(log_probs).flatten().cpu()
  legal_moves = board.get_legal_actions()
  legal_moves_indices = []
  for move in legal_moves:
    if move == "pass":
      legal_moves_indices.append(81)
    else:
      (x, y) = move
      legal_moves_indices.append(y*9+x)
  filtered_probs = torch.zeros(82)
  for index in legal_moves_indices:
    filtered_probs[index] = move_probs[index]
  ai_move = torch.multinomial(input=filtered_probs,num_samples=1).item()
  return "pass" if ai_move == 81 else (ai_move % 9, ai_move // 9)




class PolicyNetwork1(nn.Module):
  def __init__(self):
    super(PolicyNetwork1, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
    self.fc = nn.Linear(in_features=81, out_features=82)  # Fully connected layer

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.conv3(x)
    x = x.view(x.size(0), -1) # flattening
    x = self.fc(x)
    return F.log_softmax(x, dim=1)




############

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
  try:
    payload = request.get_json() # dict representing the json
    board = np.array(payload["board"])
    player = payload["player"]
    ko = None if payload["ko"] == "" else (payload["ko"].split(",")[0], payload["ko"].split(",")[1])
    ko_player = None if payload["koPlayer"] == 0 else payload["koPlayer"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_white_move = None if payload["lastWhiteMove"] == "" else (payload["lastWhiteMove"].split(",")[0], payload["lastWhiteMove"].split(",")[1])
    last_black_move = None if payload["lastBlackMove"] == "" else (payload["lastBlackMove"].split(",")[0], payload["lastBlackMove"].split(",")[1])
    model = PolicyNetwork1().to(device)
    model.load_state_dict(torch.load('policy_net_percentfilled.pth', map_location=torch.device("cpu")))
    model.eval()
    board_obj = Board(9,9)
    board_obj._board = board
    board_obj.player = int(player)
    board_obj._last_black_move = last_black_move
    board_obj._last_white_move = last_white_move
    board_obj._ko = ko
    board_obj._ko_player = ko_player
    ai_move = select_move(model, board_obj, device)
    ai_move_str = "pass" if ai_move == "pass" else str(ai_move).split(",")[0][1:].strip() +"," + str(ai_move).split(",")[1][:-1].strip()
    print("AI Suggests Moving to: " + ai_move_str)

    if not payload:
      return jsonify({"error":"No data provided"}), 400
   
    return jsonify({"move": ai_move_str}), 200
  
  except Exception as e:
    return jsonify({"error!: ": str(e)}), 500

if __name__ == "__main__":
  app.run(debug=True)