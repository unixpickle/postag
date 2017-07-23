package postag

import (
	"math"

	"github.com/unixpickle/hmm"
)

// EndTag is the terminal state for the HMM.
const EndTag = "<end>"

// Train builds a maximum-likelihood Model from the
// training data.
func Train(data []Sample) *Model {
	emitter := hmm.TabularEmitter{}
	res := &Model{
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
				emitter[tag] = map[hmm.Obs]float64{}
			}
			emitter[tag][sample.Tokens[i]]++

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
		for obs, freq := range dist {
			dist[obs] = math.Log(freq / count)
		}
	}

	return res
}
