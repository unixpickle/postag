package postag

import (
	"bytes"
	"encoding/gob"
	"io/ioutil"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/hmm"
)

func init() {
	gob.Register(hmm.TabularEmitter{})
}

// A Model is a complete POS tagger.
type Model struct {
	HMM *hmm.HMM
}

// LoadModel loads a model from a file.
func LoadModel(path string) (model *Model, err error) {
	defer essentials.AddCtxTo("load model", &err)
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	model = &Model{}
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&model.HMM); err != nil {
		return nil, err
	}
	return model, nil
}

// Tag produces tags for the tokenized phrase.
func (m *Model) Tag(tokens []string) []string {
	obsSeq := make([]hmm.Obs, len(tokens))
	for i, tok := range tokens {
		obsSeq[i] = tok
	}
	var res []string
	for _, state := range hmm.MostLikely(m.HMM, obsSeq) {
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
	return ioutil.WriteFile(path, hmmData.Bytes(), 0755)
}
