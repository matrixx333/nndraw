class Matrix:
    def __init__(self, rows: list[list[float]]):
        if len(set(len(row) for row in rows)) > 1:
            raise ValueError("All rows mut have the same number of columns")
        self._rows = rows

    def __getitem__(self, index: int) -> "Matrix":
        return self;

    @property
    def rows(self) -> int:
        return len(self._rows)
    
    @property 
    def cols(self) -> int:
        return len(self._rows[0])