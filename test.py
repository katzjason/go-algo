import torch
import matplotlib as plt
from collections import Counter
from dataset import GoChunkedDataset

def main():
    
  dataset = GoChunkedDataset("./train_data")
  values = []
  for i in range(len(dataset)):
    _, _, y_val = dataset[i]
    values.append(y_val.item())

  print(Counter(values))




if __name__ == "__main__":
    main()