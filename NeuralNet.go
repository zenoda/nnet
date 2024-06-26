package nnet

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"time"
)

type NeuralNetConfig struct {
	//各层的神经元数量
	Neurons []int
	//各层的激活函数。输入层没有。
	ActivationFuncList []func(*mat.Dense) *mat.Dense
	//各层的求导函数。输入层没有。
	DerivationFuncList []func(*mat.Dense) *mat.Dense
}

type NeuralNet struct {
	NeuralNetConfig
	//各层的输出。第一层的输出与整个网络的输入相同
	outputs []*mat.Dense
	//各层权重。注意：输出层没有权重，也就是len(weights)比层数小1
	weights []*mat.Dense
	//各层偏置权重。注意：输出层没有偏置，也就是len(biases)比层数小1
	biases []*mat.Dense
	//各层梯度
	slopes []*mat.Dense
}

func (nn NeuralNet) Train(inputs *mat.Dense, labels *mat.Dense, learnRate float64, maxEpochs int) {
	bestEpoch := 0
	minError := 1000.0
	for i := 0; i < maxEpochs; i++ {
		outputs := nn.forward(inputs, 0)
		loss := calcLoss(labels, outputs)
		totalError := calcAvgError(loss)
		fmt.Printf("Total error of epoch %d: %.15f\n", i, totalError)
		if totalError < minError {
			minError = totalError
			bestEpoch = i
		}
		nn.backward(loss, len(nn.Neurons)-1, learnRate)
	}

	fmt.Printf("Best Epoch: %d\n", bestEpoch)
	fmt.Printf("Minimum Error: %.15f\n", minError)
}

func calcAvgError(loss *mat.Dense) float64 {
	e := new(mat.Dense)
	e.MulElem(loss, loss)
	ev := 0.0
	ed := e.RawMatrix().Data
	for j := range len(ed) {
		ev += ed[j]
	}
	ev = ev / float64(len(ed))
	return ev
}

func calcLoss(labels *mat.Dense, outputs *mat.Dense) *mat.Dense {
	loss := new(mat.Dense)
	loss.Sub(labels, outputs)
	return loss
}

func (nn NeuralNet) forward(inputs *mat.Dense, layer int) *mat.Dense {
	nn.outputs[layer] = new(mat.Dense)
	if layer == 0 {
		//输入层的输出=输入
		nn.outputs[layer] = inputs
	} else {
		//其他层的输出=激活函数(输入)
		nn.outputs[layer] = nn.ActivationFuncList[layer-1](inputs)
		//计算并保存当前层的梯度数据
		nn.slopes[layer] = nn.DerivationFuncList[layer-1](inputs)
	}
	if layer < len(nn.Neurons)-1 {
		newInputs := new(mat.Dense)
		newInputs.Mul(nn.outputs[layer], nn.weights[layer])
		newInputs.Apply(nn.addBiasFunc(layer), newInputs)
		return nn.forward(newInputs, layer+1)
	}
	return nn.outputs[layer]
}

func (nn NeuralNet) backward(loss *mat.Dense, layer int, learningRate float64) {
	nextLoss := new(mat.Dense)
	if layer == len(nn.Neurons)-1 {
		nextLoss.MulElem(loss, nn.slopes[layer])
	} else {
		if layer > 0 {
			nextLoss.Mul(loss, nn.weights[layer].T())
			nextLoss.MulElem(nextLoss, nn.slopes[layer])
		}
		wAdjusts := new(mat.Dense)
		wAdjusts.Mul(nn.outputs[layer].T(), loss)
		wAdjusts.Scale(1.0/float64(loss.RawMatrix().Rows), wAdjusts)
		wAdjusts.Scale(learningRate, wAdjusts)
		//fmt.Printf("Weights: %v\n", nn.weights[layer].RawMatrix().Data)
		//fmt.Printf("Weight Adjusts: %v\n", wAdjusts.RawMatrix().Data)
		nn.weights[layer].Add(nn.weights[layer], wAdjusts)
		bAdjusts := mat.NewDense(1, loss.RawMatrix().Cols, nil)
		for i := range loss.RawMatrix().Cols {
			val := 0.0
			rowCount := loss.RawMatrix().Rows
			for j := range rowCount {
				val += loss.RawRowView(j)[i]
			}
			bAdjusts.Set(0, i, val/float64(rowCount))
		}
		bAdjusts.Scale(learningRate, bAdjusts)
		nn.biases[layer].Add(nn.biases[layer], bAdjusts)
	}
	if layer > 0 {
		nn.backward(nextLoss, layer-1, learningRate)
	}
}

func (nn NeuralNet) addBiasFunc(layer int) func(i int, j int, v float64) float64 {
	return func(i int, j int, v float64) float64 {
		return v + nn.biases[layer].RawRowView(0)[j]
	}
}

func (nn NeuralNet) Predict(inputs *mat.Dense) (outputs *mat.Dense) {
	for layer := range len(nn.Neurons) {
		if layer == 0 {
			outputs = inputs
		} else {
			outputs = nn.ActivationFuncList[layer-1](inputs)
		}
		if layer < len(nn.Neurons)-1 {
			inputs = new(mat.Dense)
			inputs.Mul(outputs, nn.weights[layer])
			inputs.Apply(nn.addBiasFunc(layer), inputs)
		}
	}
	return
}

func (nn NeuralNet) Evaluate(inputs *mat.Dense, labels *mat.Dense) (avgErr float64) {
	outputs := nn.Predict(inputs)
	loss := calcLoss(labels, outputs)
	avgErr = calcAvgError(loss)
	return
}

func NewNeuralNet(config NeuralNetConfig) *NeuralNet {
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	nn := &NeuralNet{NeuralNetConfig: config}
	layers := len(nn.Neurons)
	var outputs []*mat.Dense
	var weights []*mat.Dense
	var biases []*mat.Dense
	var slopes []*mat.Dense
	for i := 0; i < layers-1; i++ {
		rowCount := nn.Neurons[i]
		colCount := nn.Neurons[i+1]
		var wRawData []float64
		for j := 0; j < rowCount; j++ {
			for k := 0; k < colCount; k++ {
				wRawData = append(wRawData, randGen.Float64())
			}
		}
		var bRawData []float64
		for j := 0; j < colCount; j++ {
			bRawData = append(bRawData, randGen.Float64())
		}
		w := mat.NewDense(rowCount, colCount, wRawData)
		weights = append(weights, w)
		b := mat.NewDense(1, colCount, bRawData)
		biases = append(biases, b)
		slopes = append(slopes, new(mat.Dense))
		outputs = append(outputs, new(mat.Dense))
	}
	slopes = append(slopes, new(mat.Dense))
	outputs = append(outputs, new(mat.Dense))
	nn.weights = weights
	nn.biases = biases
	nn.slopes = slopes
	nn.outputs = outputs
	return nn
}

// Sigmoid 激活函数
func Sigmoid(inputs *mat.Dense) *mat.Dense {
	outputs := new(mat.Dense)
	outputs.Apply(func(i, j int, v float64) float64 {
		return 1.0 / (1.0 + math.Exp(-v))
	}, inputs)
	return outputs
}

// Relu 激活函数
func Relu(inputs *mat.Dense) *mat.Dense {
	outputs := new(mat.Dense)
	outputs.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return v
		}
		return 0
	}, inputs)
	return outputs
}

// LRelu 激活函数
func LRelu(inputs *mat.Dense) *mat.Dense {
	outputs := new(mat.Dense)
	outputs.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return v
		}
		return 0.01 * v
	}, inputs)
	return outputs
}

// Softmax 激活函数
func Softmax(inputs *mat.Dense) *mat.Dense {
	outputsRawData := inputs.RawMatrix().Data
	rows := inputs.RawMatrix().Rows
	cols := inputs.RawMatrix().Cols
	for i := range rows {
		var maxVal float64
		for j := range cols {
			data := outputsRawData[i*cols+j]
			if j == 0 {
				maxVal = data
			} else if data > maxVal {
				maxVal = data
			}
		}
		total := 0.0
		for j := range cols {
			outputsRawData[i*cols+j] = math.Exp(outputsRawData[i*cols+j] - maxVal)
			total += outputsRawData[i*cols+j]
		}
		for j := range cols {
			outputsRawData[i*cols+j] /= total
		}
	}
	outputs := mat.NewDense(rows, cols, outputsRawData)
	return outputs
}

// DSoftmax 反向传播函数
func DSoftmax(inputs *mat.Dense) *mat.Dense {
	outputs := new(mat.Dense)
	outputs.Scale(1, inputs)
	return outputs
}

// DSigmoid 反响传播函数：dSigmoid
func DSigmoid(inputs *mat.Dense) *mat.Dense {
	outputs := new(mat.Dense)
	outputs.Apply(func(i, j int, v float64) float64 {
		return (1.0 / (1.0 + math.Exp(-v))) * (1.0 - 1.0/(1.0+math.Exp(-v)))
	}, inputs)
	return outputs
}

// DRelu 反向传播函数：dRelu
func DRelu(inputs *mat.Dense) *mat.Dense {
	outputs := new(mat.Dense)
	outputs.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return 1
		}
		return 0
	}, inputs)
	return outputs
}

// DLRelu 反向传播函数
func DLRelu(inputs *mat.Dense) *mat.Dense {
	outputs := new(mat.Dense)
	outputs.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return 1
		}
		return 0.01
	}, inputs)
	return outputs
}
