package dtype

type Dtype int32

const (
	InvalidDtype Dtype = iota
	Float32
	Float64
	Int32
	Int64
	Bool
)
