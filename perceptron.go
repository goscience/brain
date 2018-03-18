package brain

import (
	"math/rand"
	"math"
	"errors"
)

type Perceptron struct {
	inputSize   int
	nodeWeights []float64
}

func NewPerceptron(inputSize int) *Perceptron {
	pr := &Perceptron{}
	pr.inputSize = inputSize

	for i := 0; i < inputSize; i++ {
		weight := 2*rand.Float64() - 1
		pr.nodeWeights = append(pr.nodeWeights, weight)
	}

	return pr
}

func (pr *Perceptron) GetNodeWeights() []float64 {
	return pr.nodeWeights
}

func Train(
	pr *Perceptron,
	trainingSet [][]float64,
	targetSet []float64,
	epoch int,
) {
	for i := 0; i < epoch; i++ {
		var derivatives []float64

		for j, dataRow := range trainingSet {
			nodes := pr.GetNodeWeights()
			weight, _ := dot(dataRow, nodes)
			normalized := sigmoid(weight)
			err := targetSet[j] - normalized
			derivatives = append(
				derivatives,
				err*sigmoidDerivative(normalized),
			)
		}

		transpose := transpose(trainingSet)

		for k, row := range transpose {
			adjustment, _ := dot(
				row,
				derivatives,
			)

			pr.nodeWeights[k] += adjustment
		}
	}
}

func Predict(pr *Perceptron, data []float64) float64 {
	vectorDot, _ := dot(data, pr.GetNodeWeights())

	return sigmoid(vectorDot)
}

func sigmoid(weight float64) float64 {
	return 1 / (1 + math.Exp(-weight))
}

func sigmoidDerivative(weight float64) float64 {
	return weight * (1 - weight)
}

func transpose(matrix [][]float64) [][]float64 {
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

func dot(x, y []float64) (r float64, err error) {
	if len(x) != len(y) {
		return 0, errors.New("incompatible lengths")
	}

	for i, xi := range x {
		r += xi * y[i]
	}

	return
}
