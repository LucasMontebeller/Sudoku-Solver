import numpy as np
import pandas as pd 

class Element():
    def __init__(self, x, y, score):
        self.x = x
        self.y = y
        self.score = score
        self.possible_values = set()

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

    def get_no_zero_columns(self):
        index_columns = dict()
        for j in range(self.grid_size[0]):
            index_columns[j] = np.count_nonzero(self.grid[:, j])

        return index_columns
    
    def get_no_zero_rows(self):
        index_rows = dict()
        for i in range(self.grid_size[0]):
            index_rows[i] = np.count_nonzero(self.grid[i])

        return index_rows
    
    def get_squares(self):
        interval_index = dict()
        grid_reshaped = self.grid_reshaped
        for i in range(grid_reshaped.ndim):
            for j in range(grid_reshaped.ndim):
                square = grid_reshaped[i * 3 + j]
                position_range = (i * 3, i * 3 + 2, j * 3, j * 3 + 2)
                interval_index[position_range] = np.count_nonzero(square)
        
        return interval_index
    
    def get_square_score_by_element(self, i, j):
        for position_range, score in self.get_squares().items():
            i_range, j_range = position_range[:2], position_range[2:]
            if i_range[0] <= i <= i_range[1] and j_range[0] <= j <= j_range[1]:
                return score

        Exception("Element not found in square interval")
            

    def get_candidates(self):
        rows = self.get_no_zero_rows()
        columns = self.get_no_zero_columns()

        index_zero_rows, index_zero_columns = np.where(self.grid == 0)
        if len(index_zero_rows) != len(index_zero_columns):
            Exception("Size of sets index_zero_rows and index_zero_columns are different.")
    
        candidates = []
        for i, j in zip(index_zero_rows, index_zero_columns):
            score = rows[i] + columns[j] + self.get_square_score_by_element(i, j) # melhorar score futuramente --> retirar poissíveis repetidos 
            element = Element(i, j, score)
            candidates.append(element)
                
        return candidates
    
    # Regras do jogo
    def get_possible_values(self, element: Element):
        len_square = self.grid_size[0] // 3
        valid_elements = {i for i in range(1, 10)}

        for i in self.grid[element.x]:
            if i != 0:
                valid_elements.remove(i)

        for j in self.grid[:, element.y]:
            if j != 0 and j in valid_elements:
                valid_elements.remove(j)

        square_index = (element.x // len_square) * 3 + (element.y // len_square)
        square = self.grid_reshaped[square_index]
        for i in range(len_square):
            for j in range(len_square):
                neighbor_element = square[i][j]
                if neighbor_element != 0 and neighbor_element in valid_elements:
                    valid_elements.remove(neighbor_element)

        if len(valid_elements) == 0:
            Exception(f'Wrong validations on the element {element.x, element.y}')

        return valid_elements
        
    def solve(self):
        total_zeros = np.power(len(self.grid), 2) - np.count_nonzero(self.grid)
        while total_zeros != 0:
            candidates = self.get_candidates()
            candidates.sort(key=lambda x: x.score, reverse=True)

            for c in candidates:
                values = self.get_possible_values(c)
                c.possible_values = values
                if len(values) == 1:
                    self.grid[c.x, c.y] = next(iter(values))
                    print(f'Find element ! {c.x, c.y} : {values}')
                    self.solve()
                    
            # incluir validação cruzada (portas lógicas)
            break



def main():
    tabuleiro = pd.read_csv('tabuleiro.csv')
    sudoku = Sudoku(tabuleiro.to_numpy(dtype=np.uint8))
    sudoku.solve()

if __name__ == '__main__':
    main()