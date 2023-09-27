import numpy as np
import pandas as pd 

# Lembrar de aplicar os principios SOLID e otimizar o código quando estiver pronto !
class Element():
    def __init__(self, x, y, score):
        self.x = x
        self.y = y
        self.score = score
        self.possible_values = set()

class Sudoku():
    def __init__(self, grid:np.ndarray):
        self.grid = grid
        self._grid_size = grid.shape
        self._grid_reshaped = self.__get_grid_reshaped()
        self.cont = 0
    
    def __get_grid_reshaped(self):
        len_square = self._grid_size[0] // 3
        return np.array([
            self.grid[i: i + len_square, j: j + len_square] 
            for i in range(0, self._grid_size[0], len_square) 
            for j in range(0, self._grid_size[0], len_square)
        ])

    def get_no_zero_columns(self):
        index_columns = dict()
        for j in range(self._grid_size[0]):
            column = self.grid[:, j]
            index_columns[j] = column[np.where(column != 0)]

        return index_columns
    
    def get_no_zero_rows(self):
        index_rows = dict()
        for i in range(self._grid_size[0]):
            row = self.grid[i]
            index_rows[i] = row[np.where(row != 0)]

        return index_rows
    
    def get_no_zero_elements_squares(self):
        interval_index = dict()
        for i in range(self._grid_reshaped.ndim):
            for j in range(self._grid_reshaped.ndim):
                square = self._grid_reshaped[i * 3 + j]
                position_range = (i * 3, i * 3 + 2, j * 3, j * 3 + 2)
                interval_index[position_range] = square[np.where(square != 0)]
        
        return interval_index
    
    def get_square_by_element(self, i, j):
        square_elements = self.get_no_zero_elements_squares().items()
        for position_range in square_elements:
            i_range, j_range = position_range[0][:2], position_range[0][2:]
            if i_range[0] <= i <= i_range[1] and j_range[0] <= j <= j_range[1]:
                return position_range[1]

        Exception("Element not found in square interval")
            

    def get_candidates(self):
        rows = self.get_no_zero_rows()
        columns = self.get_no_zero_columns()

        index_zero_rows, index_zero_columns = np.where(self.grid == 0)
        if len(index_zero_rows) != len(index_zero_columns):
            Exception("Size of sets index_zero_rows and index_zero_columns are different.")
    
        candidates = []
        for i, j in zip(index_zero_rows, index_zero_columns):
            row_column_elements = np.union1d(rows[i], columns[j])
            square_elements = self.get_square_by_element(i, j)
            score = np.union1d(row_column_elements, square_elements).size
            element = Element(i, j, score)
            candidates.append(element)
                
        return candidates
    
    # Regras do jogo
    def get_possible_values(self, element: Element):
        len_square = self._grid_size[0] // 3
        valid_elements = {i for i in range(1, 10)}

        for i in self.grid[element.x]:
            if i != 0:
                valid_elements.remove(i)

        for j in self.grid[:, element.y]:
            if j != 0 and j in valid_elements:
                valid_elements.remove(j)

        square_index = (element.x // len_square) * 3 + (element.y // len_square)
        square = self._grid_reshaped[square_index]
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
        # if total_zeros != 0:
        if self.cont < 50: # temporario para não entrar em loop infinito
            candidates = self.get_candidates()
            candidates.sort(key=lambda x: x.score, reverse=True)

            # Preenche com as opções possiveis
            candidates_found = set()
            for c in candidates:
                values = self.get_possible_values(c)
                c.possible_values = values
                if len(values) == 1:
                    candidates_found.add(c)

            # Para os que possuem somente uma opção, preenche e avalia novamente
            for c in candidates_found:
                value = next(iter(c.possible_values))
                self.grid[c.x, c.y] = value
                print(f'Element found ! Iteration: {self.cont}, Element: {c.x, c.y} = {value}')
            
            # Se não (?) Dupla solitária? validar melhor estratégia

            print(f'No elements found ! Iteration: {self.cont}')
            self.cont += 1
            self.solve()        
        
        else:
            print(f'Total remaining elements : {total_zeros}')


def main():
    tabuleiro = pd.read_csv('tabuleiro.csv')
    sudoku = Sudoku(tabuleiro.to_numpy(dtype=np.uint8))
    sudoku.solve()

if __name__ == '__main__':
    main()