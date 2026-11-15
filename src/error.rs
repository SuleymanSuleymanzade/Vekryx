//! Errors from vector validation, retrieval, and snapshot decoding.
use std::fmt;

#[derive(Debug)]
pub enum Error {
    InvalidArgument(String),
    InvalidSnapshot(String),
    Json(serde_json::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message) | Self::InvalidSnapshot(message) => {
                formatter.write_str(message)
            }
            Self::Json(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::InvalidArgument(message)
    }
}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::InvalidArgument(message.to_owned())
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
