package brain

import (
	"math/rand"

	"github.com/goscience/brain/num"
)

type Layer struct {
	nodes []float64
}

func NewLayer(neurons int, inputs int) *Layer {
	layer := &Layer{}
	combinations := neurons * inputs

	for i := 0; i < combinations; i++ {
		weight := 2*rand.Float64() - 1
		layer.nodes = append(layer.nodes, weight)
	}

	return layer
}

type Network struct {
	layers []Layer
}

func NewNetwork(layers []Layer) *Network {
	n := &Network{}

	n.layers = layers

	return n
}

func TrainNetwork(
	n *Network,
	trainingSet [][]float64,
	targetSet []float64,
	epoch int,
) {
	for i := 0; i < len(n.layers); i++ {
		var input [][]float64

		for j := 0; j < epoch; j++ {
			output := trainLayer(trainingSet, targetSet, n.layers[i])
			input = append(input, output)
		}
	}
}

func trainLayer(
	trainingSet [][]float64,
	targetSet []float64,
	layer Layer,
) []float64 {
	var derivatives []float64

	for j, dataRow := range trainingSet {
		nodes := layer.nodes
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

		layer.nodes[k] += adjustment
	}

	return layer.nodes
}

func activation(weights []float64) []float64 {
	return num.SoftMax(weights)
}
