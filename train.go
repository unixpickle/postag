package postag

import (
	"math"

	"github.com/unixpickle/hmm"
	"github.com/unixpickle/wordembed"
)

// EndTag is the terminal state for the HMM.
const EndTag = "<end>"

// Train builds a maximum-likelihood Model from the
// training data.
func Train(data []Sample, embedding wordembed.Embedding) *Model {
	emitter := GaussianEmitter{}
	res := &Model{
		Embedding: embedding,
		HMM: &hmm.HMM{
			States:        []hmm.State{EndTag},
			Emitter:       emitter,
			TerminalState: EndTag,
			Init:          map[hmm.State]float64{},
			Transitions:   map[hmm.Transition]float64{},
		},
	}
	numVisited := map[hmm.State]float64{}

	for _, sample := range data {
		res.HMM.Init[sample.Tags[0]]++
		for i, tag := range sample.Tags {
			if numVisited[tag] == 0 {
				res.HMM.States = append(res.HMM.States, tag)
			}
			numVisited[tag]++

			if _, ok := emitter[tag]; !ok {
				emitter[tag] = &Gaussian{
					Mean:   make([]float64, embedding.Dim()),
					Stddev: make([]float64, embedding.Dim()),
				}
			}
			for j, x := range embeddingVec(embedding, sample.Tokens[i]) {
				emitter[tag].Mean[j] += x
				emitter[tag].Stddev[j] += x * x
			}

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
	for state, dist := range emitter {
		count := numVisited[state]
		for i := range dist.Mean {
			dist.Mean[i] /= count
			dist.Stddev[i] /= count

			// Go from the second moment to the stddev.
			dist.Stddev[i] = math.Sqrt(dist.Stddev[i] - math.Pow(dist.Mean[i], 2))
		}
	}

	return res
}

func embeddingVec(e wordembed.Embedding, token string) []float64 {
	switch data := e.Embed(token).Data().(type) {
	case []float64:
		return data
	case []float32:
		res := make([]float64, len(data))
		for i, x := range data {
			res[i] = float64(x)
		}
		return res
	}
	panic("unsupported data type")
}
