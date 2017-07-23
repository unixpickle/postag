package postag

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/hmm"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/wordembed"
)

func init() {
	serializer.RegisterTypedDeserializer((&Gaussian{}).SerializerType(),
		DeserializeGaussian)
	serializer.RegisterTypedDeserializer((GaussianEmitter{}).SerializerType(),
		DeserializeGaussianEmitter)
}

// A Model is a complete POS tagger.
type Model struct {
	Embedding wordembed.Embedding
	HMM       *hmm.HMM
}

// LoadModel loads a model from a file.
func LoadModel(path string) (model *Model, err error) {
	defer essentials.AddCtxTo("load model", &err)
	model = &Model{}
	if err := serializer.LoadAny(path, &model.Embedding, &model.HMM); err != nil {
		return nil, err
	}
	return model, nil
}

// Tag produces tags for the tokenized phrase.
func (m *Model) Tag(tokens []string) []string {
	obsSeq := make([]hmm.Obs, len(tokens))
	for i, tok := range tokens {
		obsSeq[i] = m.Embedding.Embed(tok)
	}
	hidden := hmm.MostLikely(m.HMM, obsSeq)
	var res []string
	for _, state := range hidden {
		res = append(res, string(state.(serializer.String)))
	}
	return res
}

// Save saves the model to a file.
func (m *Model) Save(path string) (err error) {
	defer essentials.AddCtxTo("save model", &err)
	return serializer.SaveAny(path, m.Embedding, m.HMM)
}

// Gaussian is a multivariate Gaussian distribution.
type Gaussian struct {
	Mean             anyvec.Vector
	InvCovariance    *anyvec.Matrix
	SqrtCovariance   *anyvec.Matrix
	CovarianceLogDet float64
}

// DeserializeGaussian deserializes a Gaussian.
func DeserializeGaussian(d []byte) (g *Gaussian, err error) {
	defer essentials.AddCtxTo("deserialize Gaussian", &err)
	var mean, invCov, sqrtCov *anyvecsave.S
	g = &Gaussian{}
	err = serializer.DeserializeAny(d, &mean, &invCov, &sqrtCov, &g.CovarianceLogDet)
	if err != nil {
		return nil, err
	}
	n := mean.Vector.Len()
	if invCov.Vector.Len() != n*n || sqrtCov.Vector.Len() != n*n {
		return nil, errors.New("bad matrix size")
	}
	g.Mean = mean.Vector
	g.InvCovariance = &anyvec.Matrix{Rows: n, Cols: n, Data: invCov.Vector}
	g.SqrtCovariance = &anyvec.Matrix{Rows: n, Cols: n, Data: sqrtCov.Vector}
	return g, nil
}

// Sample samples from the distribution.
func (g *Gaussian) Sample(gen *rand.Rand) anyvec.Vector {
	c := g.Mean.Creator()
	noise := c.MakeVector(g.Mean.Len())
	anyvec.Rand(noise, anyvec.Normal, gen)
	return g.SqrtCovariance.Apply(noise)
}

// LogProb computes the log density of the vector.
func (g *Gaussian) LogProb(vec anyvec.Vector) float64 {
	centered := vec.Copy()
	centered.Sub(g.Mean)
	expTerm := centered.Dot(g.InvCovariance.Apply(centered))

	res := float64(vec.Len())*math.Log(2*math.Pi) + g.CovarianceLogDet
	switch expTerm := expTerm.(type) {
	case float64:
		res += expTerm
	case float32:
		res += float64(expTerm)
	}

	return -0.5 * res
}

// SerializerType returns the unique ID used to serialize
// a Gaussian with the serializer package.
func (g *Gaussian) SerializerType() string {
	return "github.com/unixpickle/postag.Gaussian"
}

// Serialize serializes the Gaussian.
func (g *Gaussian) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		&anyvecsave.S{Vector: g.Mean},
		&anyvecsave.S{Vector: g.InvCovariance.Data},
		&anyvecsave.S{Vector: g.SqrtCovariance.Data},
		g.CovarianceLogDet,
	)
}

// A GaussianEmitter is an hmm.Emitter for emitting
// continuous word embeddings according to Gaussian
// distributions.
type GaussianEmitter map[hmm.State]*Gaussian

// DeserializeGaussianEmitter deserializes a
// GaussianEmitter.
func DeserializeGaussianEmitter(d []byte) (g GaussianEmitter, err error) {
	defer essentials.AddCtxTo("deserialize GaussianEmitter", &err)
	var states, gaussians []serializer.Serializer
	if err := serializer.DeserializeAny(d, &states, &gaussians); err != nil {
		return nil, err
	}
	if len(states) != len(gaussians) {
		return nil, errors.New("length mismatch")
	}
	g = GaussianEmitter{}
	for i, state := range states {
		gaussian, ok := gaussians[i].(*Gaussian)
		if !ok {
			return nil, errors.New("not a Gaussian")
		}
		if !reflect.TypeOf(state).Comparable() {
			return nil, errors.New("state not comparable")
		}
		g[state] = gaussian
	}
	return g, nil
}

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
			res = append(res, dist.LogProb(obs.(anyvec.Vector)))
		} else {
			res = append(res, math.Inf(-1))
		}
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a GaussianEmitter with the serializer package.
func (g GaussianEmitter) SerializerType() string {
	return "github.com/unixpickle/postag.GaussianEmitter"
}

// Serialize serializes the GaussianEmitter.
func (g GaussianEmitter) Serialize() (data []byte, err error) {
	defer essentials.AddCtxTo("serialize GaussianEmitter", &err)
	var states []serializer.Serializer
	var gaussians []serializer.Serializer
	for state, gaussian := range g {
		stateSer, ok := state.(serializer.Serializer)
		if !ok {
			return nil, fmt.Errorf("not a Serializer: %T", state)
		}
		states = append(states, stateSer)
		gaussians = append(gaussians, gaussian)
	}
	return serializer.SerializeAny(states, gaussians)
}
