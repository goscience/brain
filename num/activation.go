package num

import "math"

func Sigmoid(weight float64) float64 {
	return 1 / (1 + math.Exp(-weight))
}

func SigmoidDerivative(weight float64) float64 {
	return weight * (1 - weight)
}

func SoftMax(x []float64) []float64 {
	max := x[0]

	for _, n := range x {
		max = math.Max(max, n)
	}

	a := make([]float64, len(x))

	sum := 0.0

	for i, n := range x {
		a[i] -= math.Exp(n - max)
		sum += a[i]
	}

	for i, n := range a {
		a[i] = n / sum
	}

	return a
}
