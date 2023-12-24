import numpy as np
import pandas as pd 
import time

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
        self.cont = 0
    
    def get_grid_reshaped(self):
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
        grid_reshaped = self.get_grid_reshaped()
        for i in range(grid_reshaped.ndim):
            for j in range(grid_reshaped.ndim):
                square = grid_reshaped[i * 3 + j]
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
        grid_reshaped = self.get_grid_reshaped()
        square = grid_reshaped[square_index]
        for i in range(len_square):
            for j in range(len_square):
                neighbor_element = square[i][j]
                if neighbor_element != 0 and neighbor_element in valid_elements:
                    valid_elements.remove(neighbor_element)

        if len(valid_elements) == 0:
            Exception(f'Wrong validations on the element {element.x, element.y}')

        return valid_elements
    
    def get_by_exclusion(self, candidates):
        elements = self.get_two_possible_elements(candidates)
        pair = []
        for index_i, i in enumerate(elements):
            for j in elements[index_i + 1:]:
                if i.possible_values == j.possible_values and (self.is_same_line_or_column(i, j) or self.is_neighbor(i, j)):
                    pair.append((i, j)) 
        return pair

    def get_two_possible_elements(self, candidates):
        filtered = filter(lambda x: len(x.possible_values) == 2, candidates)
        return list(filtered)
    
    def is_neighbor(self, element_a: Element, element_b: Element):
        square_a = self.get_square_by_element(element_a.x, element_a.y)
        square_b = self.get_square_by_element(element_b.x, element_b.y)
        return True if np.array_equal(square_a, square_b) else False
    
    def is_same_line_or_column(self, element_a: Element, element_b: Element):
        return True if element_a.x == element_b.x or element_a.y == element_b.y else False
    
    def restricts_possibilities(self, candidates, elements:tuple):
        element_a, element_b = elements
        possible_values = element_a.possible_values
        filtered = []

        if element_a.x == element_b.x:
            filtered = filter(lambda c: c.x == element_a.x, candidates)
        elif element_a.y == element_b.y:
            filtered = filter(lambda c: c.y == element_a.y, candidates)

        filtered_candidates = set(filtered)
        if self.is_neighbor(element_a, element_b):
            square = self.get_square_by_element(element_a.x, element_a.y)
            new_filtered = filter(lambda c: np.array_equal(self.get_square_by_element(c.x, c.y), square), candidates)
            new_filtered_candidates = set(new_filtered)
            filtered_candidates.union(new_filtered_candidates)        
        
        for c in filtered_candidates:
            if c not in elements:
                c.possible_values.difference_update(possible_values)

        return candidates
    
    def restricts_square_possibilities(self, candidates, element:Element):
        possible_values = element.possible_values
        square = self.get_square_by_element(element.x, element.y)
        filtered = filter(lambda c: np.array_equal(self.get_square_by_element(c.x, c.y), square), candidates)
        same_square = set(filtered).difference({element})

        possible_values_square = set()
        for c in same_square:
            values = c.possible_values
            for v in values:
                possible_values_square.add(v)

        only_value = possible_values.difference(possible_values_square)
        if len(only_value) != 0:
            element.possible_values = only_value

        return candidates

    def solve(self, candidates=None):
        total_zeros = np.power(len(self.grid), 2) - np.count_nonzero(self.grid)
        if total_zeros == 0:
            print()
            print('Sudoku solved !\n')
            print(self.grid)
            return

        if self.cont < 50: # temporario para não entrar em loop infinito
            candidates = self.get_candidates() if candidates is None else candidates
            candidates.sort(key=lambda x: x.score, reverse=True)

            # Preenche com as opções possiveis
            candidates_found = set()
            for c in candidates:
                values = self.get_possible_values(c)
                c.possible_values = values
                if len(values) == 1:
                    candidates_found.add(c)

            # Implementar validação por exclusão
            # Se existe na mesma linha, coluna ou quadrante dois elementos com um par de possibilidade {z, k}, 
            # logo nenhum outro elemento nessas mesmas condições poderá ter o valor.
            if candidates_found is None or len(candidates_found) == 0:
                exclusion_elements = self.get_by_exclusion(candidates)
                if len(exclusion_elements) == 0:
                    Exception('No elements found !')
                    exit()

                for i in exclusion_elements:
                    candidates = self.restricts_possibilities(candidates, i)

                # Verifica se apenas o elemento pode assumir aquele valor
                for i, j in exclusion_elements:
                    candidates = self.restricts_square_possibilities(candidates, i)
                    candidates = self.restricts_square_possibilities(candidates, j)

                filtered = filter(lambda c: len(c.possible_values) == 1, candidates)
                new_candidates_found = list(filtered)
                candidates_found = candidates_found.union(set(new_candidates_found))

            # Para os que possuem somente uma opção, preenche e avalia novamente
            for c in candidates_found:
                value = next(iter(c.possible_values))
                self.grid[c.x, c.y] = value
                print(f'Element found ! Iteration: {self.cont}, Element: {c.x, c.y} = {value}')
            
            self.cont += 1
            candidates = set(candidates).difference(candidates_found)
            self.solve(list(candidates))        


def main():
    tabuleiro = pd.read_csv('tabuleiro_011.csv')
    sudoku = Sudoku(tabuleiro.to_numpy(dtype=np.uint8))
    start = time.time()
    sudoku.solve()
    end = time.time()
    print(f'Iterations : {sudoku.cont}, Time : {end - start}')

if __name__ == '__main__':
    main()