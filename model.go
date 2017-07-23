package postag

import (
	"bytes"
	"encoding/gob"
	"io/ioutil"
	"math"
	"math/rand"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/hmm"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/wordembed"
)

func init() {
	gob.Register(GaussianEmitter{})
}

// A Model is a complete POS tagger.
type Model struct {
	Embedding wordembed.Embedding
	HMM       *hmm.HMM
}

// LoadModel loads a model from a file.
func LoadModel(path string) (model *Model, err error) {
	defer essentials.AddCtxTo("load model", &err)
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	model = &Model{}
	var hmmData []byte
	if err := serializer.DeserializeAny(data, &model.Embedding, &hmmData); err != nil {
		return nil, err
	}
	if err := gob.NewDecoder(bytes.NewReader(hmmData)).Decode(&model.HMM); err != nil {
		return nil, err
	}
	return model, nil
}

// Tag produces tags for the tokenized phrase.
func (m *Model) Tag(tokens []string) []string {
	obsSeq := make([]hmm.Obs, len(tokens))
	for i, tok := range tokens {
		obsSeq[i] = embeddingVec(m.Embedding, tok)
	}
	hidden := hmm.MostLikely(m.HMM, obsSeq)
	var res []string
	for _, state := range hidden {
		res = append(res, state.(string))
	}
	return res
}

// Save saves the model to a file.
func (m *Model) Save(path string) (err error) {
	defer essentials.AddCtxTo("save model", &err)
	var hmmData bytes.Buffer
	if err := gob.NewEncoder(&hmmData).Encode(m.HMM); err != nil {
		return err
	}
	data, err := serializer.SerializeAny(m.Embedding, hmmData.Bytes())
	if err != nil {
		return err
	}
	return ioutil.WriteFile(path, data, 0755)
}

// Gaussian represents a diagonal multivariate gaussian.
type Gaussian struct {
	Mean   []float64
	Stddev []float64
}

// Sample samples from the distribution.
func (g *Gaussian) Sample(gen *rand.Rand) []float64 {
	res := make([]float64, len(g.Mean))
	for i, mean := range g.Mean {
		if gen == nil {
			res[i] = rand.NormFloat64()
		} else {
			res[i] = gen.NormFloat64()
		}
		res[i] = mean + g.Stddev[i]*res[i]
	}
	return res
}

// LogProb computes the log density of the vector.
func (g *Gaussian) LogProb(vec []float64) float64 {
	var res float64
	for i, comp := range vec {
		variance := g.Stddev[i] * g.Stddev[i]
		res += math.Log(1 / math.Sqrt(2*math.Pi*variance))
		res -= math.Pow(comp-g.Mean[i], 2) / (2 * variance)
	}
	return res
}

// A GaussianEmitter is an hmm.Emitter for emitting
// continuous word embeddings according to Gaussian
// distributions.
type GaussianEmitter map[hmm.State]*Gaussian

// Sample samples a []float64 from the distribution.
func (g GaussianEmitter) Sample(gen *rand.Rand, state hmm.State) hmm.Obs {
	return g[state].Sample(gen)
}

// LogProbs computes conditional densities for a []float64
// observation.
func (g GaussianEmitter) LogProbs(obs hmm.Obs, states ...hmm.State) []float64 {
	var res []float64
	for _, state := range states {
		if dist, ok := g[state]; ok {
			res = append(res, dist.LogProb(obs.([]float64)))
		} else {
			res = append(res, math.Inf(-1))
		}
	}
	return res
}
