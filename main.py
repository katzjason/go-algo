import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from flask import jsonify, request, Flask
import json
from flask_cors import CORS

from engine import Board, board_to_tensor
from model import ResNet


def select_move(policy_net, board, device="cpu", temperature=1.0):
  state = board_to_tensor(board)
  policy_logits, value = policy_net(state) # value is only used for training purposes
  log_probs = policy_logits / temperature
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


app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
  try:
    payload = request.get_json() # dict representing the json
    print(payload)
    board = np.array(payload["board"])
    player = payload["move_player"]
    ko = None if int(payload["ko_x"]) > 8 else (payload["ko_x"], payload["ko_y"])
    ko_player = None if payload["ko_player_restriction"] == 0 else payload["ko_player_restriction"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # LAST MOVES
    last_white_move = (payload["last_white_move"].split(",")[0],payload["last_white_move"].split(",")[1]) if "," in payload["last_white_move"] else "pass"
    last_black_move = (payload["last_black_move"].split(",")[0],payload["last_black_move"].split(",")[1]) if "," in payload["last_black_move"] else "pass"
    model = ResNet().to(device)
    model.load_state_dict(torch.load("checkpoints/go_resnet_final.pt", map_location=device))
    model.eval()
    board_obj = Board(9,9)
    # BOARD
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
  #  app.run(debug=True)
  app.run(host="0.0.0.0", port=5000)