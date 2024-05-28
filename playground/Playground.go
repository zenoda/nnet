package main

import (
	"fmt"
	. "github.com/zenoda/nnet"
	"gonum.org/v1/gonum/mat"
)

func main() {
	nn := NewNeuralNet(NeuralNetConfig{
		Neurons:            []int{2, 5, 2},
		ActivationFuncList: []func(int, int, float64) float64{Relu, Relu},
		DerivationFuncList: []func(int, int, float64) float64{DRelu, DRelu},
	})

	inputs := mat.NewDense(5, 2, []float64{0.1, 0.2, 0.3, 0.1, 0.21, 0.2, 0.51, 0.24, 0.8, 0.1})
	labels := mat.NewDense(5, 2, []float64{0.3, 1, 0.4, 1, 0.41, 1, 0.75, 1, 0.9, 1})
	nn.Train(inputs, labels, 0.1, 30000)
	fmt.Println()
	//nn.evaluate(inputs, labels)
	//nn.save()
	outputs := nn.Predict(mat.NewDense(3, 2, []float64{0.1, 0.2, 0.1, 0.1, 0.23, 0.45}))
	fmt.Println(outputs.RawMatrix().Data)
}
