package module

import "github.com/sebffischer/backend/backend/function"

type Module interface {
	// Create a new function in the module.
	// Returns an error if the function already exists or the name is invalid.
	NewFunction(name string, opts map[string]any) (function.Function, error)
}
