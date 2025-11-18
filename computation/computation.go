package computation

import "github.com/sebffischer/backend/module"

type Computation interface {
	Run(...module.ArrayValue) ([]module.ArrayValue, error)
}
