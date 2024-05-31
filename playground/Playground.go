package main

import (
	"fmt"
	. "github.com/zenoda/nnet"
	"gonum.org/v1/gonum/mat"
)

func main() {
	nn := NewNeuralNet(NeuralNetConfig{
		Neurons:            []int{2, 3, 2},
		ActivationFuncList: []func(*mat.Dense) *mat.Dense{LRelu, Softmax},
		DerivationFuncList: []func(*mat.Dense) *mat.Dense{DLRelu, DSoftmax},
	})

	inputs := mat.NewDense(5, 2, []float64{0.1, 0.2, 0.3, 0.1, 0.21, 0.2, 0.51, 0.24, 0.8, 0.1})
	labels := mat.NewDense(5, 2, []float64{0, 1, 0, 1, 0, 1, 1, 0, 1, 0})
	nn.Train(inputs, labels, 0.1, 1000000)
	fmt.Println()
	avgErr := nn.Evaluate(inputs, labels)
	fmt.Printf("Average error rate: %f\n", avgErr)
	//nn.save()
	outputs := nn.Predict(mat.NewDense(3, 2, []float64{0.1, 0.2, 0.1, 0.1, 0.8, 0.1}))
	fmt.Println(outputs.RawMatrix().Data)
}
