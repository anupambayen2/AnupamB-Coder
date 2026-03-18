# train.py  —  run as: python train.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.training.trainer import train
if __name__ == "__main__":
    train()
