package atype

import "github.com/sebffischer/backend/dtype"

type ArrayType struct {
	Dtype dtype.Dtype
	Axes  []int
}

// NumAxes returns the number of axes.
func (a *ArrayType) NumAxes() int {
	return len(a.Axes)
}
