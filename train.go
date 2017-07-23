package postag

import (
	"math"
	"runtime"
	"sync"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/hmm"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/num-analysis/linalg/eigen"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/wordembed"
)

// EndTag is the terminal state for the HMM.
const EndTag = serializer.String("<end>")

const minimumVariance = 1e-4

// Train builds a maximum-likelihood Model from the
// training data.
func Train(data []Sample, embedding wordembed.Embedding) *Model {
	res := &Model{
		Embedding: embedding,
		HMM: &hmm.HMM{
			States:        []hmm.State{EndTag},
			TerminalState: EndTag,
			Init:          map[hmm.State]float64{},
			Transitions:   map[hmm.Transition]float64{},
		},
	}
	builders := map[hmm.State]*gaussianBuilder{}
	numVisited := map[hmm.State]float64{}

	for _, sample := range data {
		res.HMM.Init[sample.Tags[0]]++
		for i, tag := range sample.Tags {
			if numVisited[tag] == 0 {
				res.HMM.States = append(res.HMM.States, tag)
			}
			numVisited[tag]++

			if _, ok := builders[tag]; !ok {
				builders[tag] = newGaussianBuilder(embedding.Dim())
			}
			builders[tag].Update(embedding.Embed(sample.Tokens[i]))

			if i > 0 {
				trans := hmm.Transition{
					From: sample.Tags[i-1],
					To:   tag,
				}
				res.HMM.Transitions[trans]++
			}
		}
		finalTrans := hmm.Transition{
			From: sample.Tags[len(sample.Tags)-1],
			To:   EndTag,
		}
		res.HMM.Transitions[finalTrans]++
	}

	for state, freq := range res.HMM.Init {
		res.HMM.Init[state] = math.Log(freq / float64(len(data)))
	}
	for trans, freq := range res.HMM.Transitions {
		res.HMM.Transitions[trans] = math.Log(freq / numVisited[trans.From])
	}

	res.HMM.Emitter = buildEmitter(builders)

	return res
}

func buildEmitter(builders map[hmm.State]*gaussianBuilder) GaussianEmitter {
	var resLock sync.Mutex
	res := GaussianEmitter{}

	inChan := make(chan *stateBuilderPair, len(builders))
	for s, b := range builders {
		inChan <- &stateBuilderPair{State: s, Builder: b}
	}
	close(inChan)

	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			for pair := range inChan {
				gaussian := pair.Builder.Gaussian()
				resLock.Lock()
				res[pair.State] = gaussian
				resLock.Unlock()
			}
			wg.Done()
		}()
	}
	wg.Wait()

	return res
}

type stateBuilderPair struct {
	State   hmm.State
	Builder *gaussianBuilder
}

type gaussianBuilder struct {
	c            anyvec.Creator
	firstMoment  linalg.Vector
	secondMoment *linalg.Matrix
	divisor      float64
}

func newGaussianBuilder(dim int) *gaussianBuilder {
	return &gaussianBuilder{
		firstMoment:  make(linalg.Vector, dim),
		secondMoment: linalg.NewMatrix(dim, dim),
	}
}

func (g *gaussianBuilder) Update(vec anyvec.Vector) {
	g.divisor++
	g.c = vec.Creator()

	v := make(linalg.Vector, vec.Len())
	switch data := vec.Data().(type) {
	case []float64:
		copy(v, data)
	case []float32:
		for i, x := range data {
			v[i] = float64(x)
		}
	}

	colMat := linalg.NewMatrixColumn(v)
	g.secondMoment.Add(colMat.Mul(colMat.Transpose()))
	g.firstMoment.Add(v)
}

func (g *gaussianBuilder) Gaussian() *Gaussian {
	g.firstMoment.Scale(1 / g.divisor)
	g.secondMoment.Scale(1 / g.divisor)

	// Turn second moment into covariance.
	firstCol := linalg.NewMatrixColumn(g.firstMoment)
	sqFirst := firstCol.Mul(firstCol.Transpose())
	g.secondMoment.Add(sqFirst.Scale(-1))

	s, lambda, sInv := g.decomposeCovariance()
	invLambda := lambda.Copy()
	sqrtLambda := lambda.Copy()
	var logDet float64
	for i := 0; i < lambda.Rows; i++ {
		logDet += math.Log(lambda.Get(i, i))
		invLambda.Set(i, i, 1/invLambda.Get(i, i))
		sqrtLambda.Set(i, i, 1/sqrtLambda.Get(i, i))
	}
	invMat := s.Mul(invLambda).Mul(sInv)
	sqrtMat := s.Mul(sqrtLambda).Mul(sInv)

	return &Gaussian{
		Mean:             g.c.MakeVectorData(g.c.MakeNumericList(g.firstMoment)),
		InvCovariance:    g.anyvecMatrix(invMat),
		SqrtCovariance:   g.anyvecMatrix(sqrtMat),
		CovarianceLogDet: logDet,
	}
}

func (g *gaussianBuilder) decomposeCovariance() (s, lambda, sInv *linalg.Matrix) {
	eigVals, eigVecs := eigen.Symmetric(g.secondMoment)
	lambda = linalg.NewMatrix(len(eigVals), len(eigVals))
	sInv = &linalg.Matrix{Rows: len(eigVals), Cols: len(eigVals)}
	for i, val := range eigVals {
		if math.Abs(val) < minimumVariance {
			val = minimumVariance
		}
		lambda.Set(i, i, val)
		sInv.Data = append(sInv.Data, eigVecs[i]...)
	}
	s = sInv.Transpose()
	return
}

func (g *gaussianBuilder) anyvecMatrix(m *linalg.Matrix) *anyvec.Matrix {
	return &anyvec.Matrix{
		Data: g.c.MakeVectorData(g.c.MakeNumericList(m.Data)),
		Rows: m.Rows,
		Cols: m.Cols,
	}
}
