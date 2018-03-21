package brain

import "math/rand"

type Layer struct {
	nodes []float64
}

func NewLayer(neurons int, inputs int) *Layer {
	layer := &Layer{}
	combinations := neurons * inputs

	for i := 0; i < combinations; i++ {
		weight := 2 * rand.Float64() - 1
		layer.nodes = append(layer.nodes, weight)
	}

	return layer
}

type Network struct {

}

func NewNetwork(layers []Layer) *Network {
	return &Network{}
}


