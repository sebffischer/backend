package atype

import (
	"fmt"

	"github.com/pkg/errors"
	"github.com/sebffischer/backend/backend/dtype"
)

// UncheckedAxis can be used in CheckDims or AssertDims functions for an axis
// whose dimension doesn't matter.
// TODO: Probably remove this once we have proper mechanism for dynamic axes.
const UncheckedAxis = int(-1)

// HasShape is an interface for objects that have an associated ArrayType.
// `tensor.Tensor` (concrete tensor) and `graph.Node` (tensor representations in a
// computation graph), `context.Variable` and Shape itself implement the interface.
// TODO(rename): Rename to HasArrayType / ArrayLike
type HasShape interface {
	ArrayType() ArrayType
}

// CheckDims checks that the shape has the given dimensions and rank. A value of -1 in
// dimensions means it can take any value and is not checked.
//
// It returns an error if the rank is different or if any of the dimensions don't match.
// TODO(rename):  CheckDims -> CheckAxesSizes, dimensions -> axes_sizes
func (s ArrayType) CheckDims(dimensions ...int) error {
	if s.Rank() != len(dimensions) {
		return errors.Errorf("shape (%s) has incompatible rank %d (wanted %d)", s, s.Rank(), len(dimensions))
	}
	for ii, wantDim := range dimensions {
		if wantDim != -1 && s.Dimensions[ii] != wantDim {
			return errors.Errorf("shape (%s) axis %d has dimension %d, wanted %d (shape wanted=%v)", s, ii, s.Dimensions[ii], wantDim, dimensions)
		}
	}
	return nil
}

// Check that the shape has the given dtype, dimensions and rank. A value of -1 in
// dimensions means it can take any value and is not checked.
//
// It returns an error if the dtype or rank is different or if any of the dimensions don't match.
// TODO(rename):  dimensions -> axes_sizes
func (s ArrayType) Check(dtype dtype.DType, dimensions ...int) error {
	if dtype != s.DType {
		return errors.Errorf("shape (%s) has incompatible dtype %s (wanted %s)", s, s.DType, dtype)
	}
	return s.CheckDims(dimensions...)
}

// AssertDims checks that the shape has the given dimensions and rank. A value of -1 in
// dimensions means it can take any value and is not checked.
//
// It panics if it doesn't match.
//
// See usage example in package shapes documentation.
// TODO(rename):  AssertDims -> AssertAxesSizes, dimensions -> axes_sizes
func (s ArrayType) AssertDims(dimensions ...int) {
	err := s.CheckDims(dimensions...)
	if err != nil {
		panic(fmt.Sprintf("atype.AssertDims(%v): %+v", dimensions, err))
	}
}

// Assert checks that the shape has the given dtype, dimensions and rank. A value of -1 in
// dimensions means it can take any value and is not checked.
//
// It panics if it doesn't match.
// TODO(rename):  Assert -> AssertAxesSizes, dimensions -> axes_sizes
func (s ArrayType) Assert(dtype dtype.DType, dimensions ...int) {
	err := s.Check(dtype, dimensions...)
	if err != nil {
		panic(fmt.Sprintf("atype.Assert(%s, %v): %+v", dtype, dimensions, err))
	}
}

// CheckDims checks that the shape has the given dimensions and rank. A value of -1 in
// dimensions means it can take any value and is not checked.
//
// It returns an error if the rank is different or any of the dimensions.
// TODO(rename):  CheckDims -> CheckAxesSizes, dimensions -> axes_sizes
func CheckDims(shaped HasShape, dimensions ...int) error {
	return shaped.ArrayType().CheckDims(dimensions...)
}

// AssertDims checks that the shape has the given dimensions and rank. A value of -1 in
// dimensions means it can take any value and is not checked.
//
// It panics if it doesn't match.
//
// See usage example in package shapes documentation.
// TODO(rename):  AssertDims -> AssertAxesSizes, dimensions -> axes_sizes
func AssertDims(shaped HasShape, dimensions ...int) {
	shaped.ArrayType().AssertDims(dimensions...)
}

// Assert checks that the shape has the given dtype, dimensions and rank. A value of -1 in
// dimensions means it can take any value and is not checked.
//
// It panics if it doesn't match.
// TODO(rename):  dimensions -> axes_sizes
func Assert(shaped HasShape, dtype dtype.DType, dimensions ...int) {
	shaped.ArrayType().Assert(dtype, dimensions...)
}

// CheckRank checks that the shape has the given rank.
//
// It returns an error if the rank is different.
// TODO(rename):  CheckRank -> CheckNumAxes, rank -> num_axes
func (s ArrayType) CheckRank(rank int) error {
	if s.Rank() != rank {
		return errors.Errorf("shape (%s) has incompatible rank %d -- wanted %d", s, s.Rank(), rank)
	}
	return nil
}

// AssertRank checks that the shape has the given rank.
//
// It panics if it doesn't match.
//
// See usage example in package shapes documentation.
// TODO(rename):  AssertRank -> AssertNumAxes, rank -> num_axes
func (s ArrayType) AssertRank(rank int) {
	err := s.CheckRank(rank)
	if err != nil {
		panic(fmt.Sprintf("assertRank(%d): %+v", rank, err))
	}
}

// CheckRank checks that the shape has the given rank.
//
// It returns an error if the rank is different.
// TODO(rename):  CheckRank -> CheckNumAxes, rank -> num_axes
func CheckRank(shaped HasShape, rank int) error {
	return shaped.ArrayType().CheckRank(rank)
}

// AssertRank checks that the shape has the given rank.
//
// It panics if it doesn't match.
//
// See usage example in package shapes documentation.
// TODO(rename):  AssertRank -> AsserNumAxes
func AssertRank(shaped HasShape, rank int) {
	shaped.ArrayType().AssertRank(rank)
}

// CheckScalar checks that the shape is a scalar.
//
// It returns an error if shape is not a scalar.
func (s ArrayType) CheckScalar() error {
	if !s.IsScalar() {
		return errors.Errorf("shape (%s) is not a scalar", s)
	}
	return nil
}

// AssertScalar checks that the shape is a scalar.
//
// It panics if it doesn't match.
//
// See usage example in package shapes documentation.
func (s ArrayType) AssertScalar() {
	err := s.CheckScalar()
	if err != nil {
		panic(fmt.Sprintf("AssertScalar(): %+v", err))
	}
}

// CheckScalar checks that the shape is a scalar.
//
// It returns an error if shape is not a scalar.
func CheckScalar(shaped HasShape) error {
	return shaped.ArrayType().CheckScalar()
}

// AssertScalar checks that the shape is a scalar.
//
// It panics if it doesn't match.
//
// See usage example in package shapes documentation.
func AssertScalar(shaped HasShape) {
	shaped.ArrayType().AssertScalar()
}
