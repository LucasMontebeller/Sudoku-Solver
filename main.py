import numpy as np
import pandas as pd 

class Sudoku():
    def __init__(self, grid:np.ndarray):
        self.grid = grid

    @property
    def grid_size(self):
        return self.grid.shape
    
    @property
    def grid_reshaped(self):
        len_square = self.grid_size[0] // 3
        return np.array([
            self.grid[i: i + len_square, j: j + len_square] 
            for i in range(0, self.grid.shape[0], len_square) 
            for j in range(0, self.grid.shape[0], len_square)
        ])

    def best_columns(self):
        index_columns = dict()
        for j in range(self.grid_size[0]):
            index_columns[j] = np.count_nonzero(self.grid[:, j])

        return index_columns
    
    def best_rows(self):
        index_rows = dict()
        for i in range(self.grid_size[0]):
            index_rows[i] = np.count_nonzero(self.grid[i])

        return index_rows
    
    def best_squares(self):
        interval_index = dict()
        grid_reshaped = self.grid_reshaped
        for i in range(grid_reshaped.ndim):
            for j in range(grid_reshaped.ndim):
                square = grid_reshaped[i * 3 + j]
                position_range = (i * 3, i * 3 + 2, j * 3, j * 3 + 2)
                interval_index[position_range] = np.count_nonzero(square)
        
        return interval_index
    
    def element_square_score(self, i, j):
        for position_range, score in self.best_squares().items():
            i_range, j_range = position_range[:2], position_range[2:]
            if i_range[0] <= i <= i_range[1] and j_range[0] <= j <= j_range[1]:
                return score

        Exception("Element not found in square interval")
            

    def choose_element(self):
       rows = self.best_rows()
       columns = self.best_columns()

       index_zero_rows, index_zero_columns = np.where(self.grid == 0)
       candidates = dict()
       for i in index_zero_rows:
           for j in index_zero_columns:
            score = rows[i] + columns[j] + self.element_square_score(i, j)
            candidates[i, j] = score
            
       return candidates

    def solve(self):
        pass

def main():
    tabuleiro = pd.read_csv('tabuleiro.csv')
    sudoku = Sudoku(tabuleiro.to_numpy(dtype=np.uint8))
    sudoku.choose_element()
    print(sudoku.grid)

if __name__ == '__main__':
    main()