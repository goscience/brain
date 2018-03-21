package brain

import (
	"math/rand"

	"github.com/goscience/brain/num"
)

type Perceptron struct {
	inputSize   int
	nodeWeights []float64
}

func NewPerceptron(inputSize int) *Perceptron {
	pr := &Perceptron{}
	pr.inputSize = inputSize

	for i := 0; i < inputSize; i++ {
		weight := 2 * rand.Float64() - 1
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
			weight, _ := num.Dot(dataRow, nodes)
			normalized := num.Sigmoid(weight)
			err := targetSet[j] - normalized
			derivatives = append(
				derivatives,
				err*num.SigmoidDerivative(normalized),
			)
		}

		transpose := num.Transpose(trainingSet)

		for k, row := range transpose {
			adjustment, _ := num.Dot(
				row,
				derivatives,
			)

			pr.nodeWeights[k] += adjustment
		}
	}
}

func Predict(pr *Perceptron, data []float64) float64 {
	vectorDot, _ := num.Dot(data, pr.GetNodeWeights())

	return num.Sigmoid(vectorDot)
}
