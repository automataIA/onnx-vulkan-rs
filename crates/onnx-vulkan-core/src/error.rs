//! ONNX core errors, independent of frontend and backend.

/// Error raised during graph validation or transformation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    Backend(String),
    InvalidShape(String),
    InvalidTensor(String),
    /// A node the interpreter cannot run. Distinct from an invalid graph: the
    /// graph is well-formed, this engine just does not cover it.
    Unsupported(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(message) => write!(formatter, "Vulkan backend error: {message}"),
            Self::InvalidShape(message) => write!(formatter, "invalid ONNX shape: {message}"),
            Self::InvalidTensor(message) => write!(formatter, "invalid ONNX tensor: {message}"),
            Self::Unsupported(message) => write!(formatter, "unsupported ONNX graph: {message}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
