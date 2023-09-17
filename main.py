import numpy as np
import pandas as pd 

class Sudoku():
    def __init__(self, grid:np.ndarray):
        self.grid = grid

    def choose_element(self):
        pass

    def solve(self):
        pass

def main():
    tabuleiro = pd.read_csv('tabuleiro.csv')
    sudoku = Sudoku(tabuleiro.to_numpy(dtype=np.uint8))
    print(sudoku.grid)

if __name__ == '__main__':
    main()