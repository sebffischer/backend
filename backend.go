package backend

import (
	"github.com/sebffischer/backend/dtype"
	"github.com/sebffischer/backend/op"
	"github.com/sebffischer/backend/platform"
)

type Backend interface {
	// Capabilities returns the capabilities of the backend:
	// - Supported operations
	// - Supported platforms
	// - Supported data types
	// - Whether to support arrays with dynamic axes
	Capabilities() (map[op.OpType]bool, map[platform.Platform]bool, map[dtype.Dtype]bool, bool)
}
