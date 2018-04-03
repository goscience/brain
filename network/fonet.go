package network

import (
	"fmt"
	"log"

	"github.com/Fontinalis/fonet"
)

func Run() {
	trainingSet := [][][]float64{
		{
			{100, 0, 100},
			{1, 1},
		},
		{
			{22, 100, 25},
			{0, 0},
		},
		{
			{2, -1, 3},
			{1, 1},
		},
	}

	n, err := fonet.NewNetwork([]int{3, 10, 10, 2})
	if err != nil {
		log.Fatal(err)
	}
	n.Train(trainingSet, 10000, 0.7, false)
	fmt.Println(n.Predict([]float64{10, 5, 10}))
}
