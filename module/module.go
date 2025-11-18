package module

import (
	"github.com/sebffischer/backend"
	"github.com/sebffischer/backend/atype"
	"github.com/sebffischer/backend/axes"
	"github.com/sebffischer/backend/dtype"
)

// All of this needs to be in the same file, because of cyclic dependencies.

// A module is a collection of one or more functions that are possibly calling one another.
type Module interface {
	// Backend returns the backend that holds this module.
	Backend() backend.Backend

	// NewFunction creates a new function with the given name.
	// This might fail if the function name is invalid or already defined.
	NewFunction(name string) (Function, error)
}

// Function is an interface for a module function.
type Function interface {
	// Module returns the module that contains this function.
	Module() Module
	// NewParameter creates a new parameter with the given data type and shape.
	NewParameter(dtype dtype.Dtype, axes axes.Axes) ArrayValue
	// Return sets the return values of the function.
	Return(...ArrayValue) Function
	// This should evaluate to true after calling Return.
	IsFinalized() bool

	// Available operations
	Add(lhs, rhs ArrayValue) (ArrayValue, error)
	Constant(data any, arrayType atype.ArrayType) (ArrayValue, error)
}

// Value is an interface for function values
type Value interface {
	// Function returns the function that contains this value.
	Function() Function
}

type ArrayValue interface {
	Value
	// Atype returns the array type of this value.
	Atype() (atype.ArrayType, error)
}
