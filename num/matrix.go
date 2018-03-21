package num

import "errors"

func Transpose(matrix [][]float64) [][]float64 {
	var transpose [][]float64

	lenX := len(matrix[0])

	for x := 0; x < lenX; x++ {
		var row []float64

		for y := range matrix {
			row = append(row, matrix[y][x])
		}

		transpose = append(transpose, row)
	}

	return transpose
}

func Dot(x, y []float64) (r float64, err error) {
	if len(x) != len(y) {
		return 0, errors.New("incompatible lengths")
	}

	for i, xi := range x {
		r += xi * y[i]
	}

	return
}
