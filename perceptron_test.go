package brain

import (
	"testing"
	"fmt"

	"github.com/stretchr/testify/assert"
)

func TestLearn(t *testing.T) {

	trainingSet := [][]float64{
		{0, 0, 1},
		{1, 1, 1},
		{1, 0, 1},
		{0, 1, 1},
	}

	targetSet := []float64{0, 1, 1, 0}

	pr := NewPerceptron(3)

	Train(pr, trainingSet, targetSet, 10000)

	result := Predict(pr, []float64{1, 1, 0})

	assert.True(t, result >= 0.9)

	fmt.Println(result)
}
