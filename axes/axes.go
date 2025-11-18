package axes

import "errors"

// Axis represents a single dimension which can be known or unknown.
type Axis struct {
	size  uint
	known bool
}

// Axes represents the shape of an array, where each dimension can be known or unknown.
type Axes []Axis

func (a Axes) NumAxes() int {
	return len(a)
}

func (a Axis) Size() (uint, error) {
	if !a.known {
		return 0, errors.New("axis size is unknown")
	}
	return a.size, nil
}

func (a Axis) Known() bool {
	return a.known
}

func (a Axes) Known() bool {
	for _, axis := range a {
		if !axis.known {
			return false
		}
	}
	return true
}
