pub mod plan;
pub mod stage;

pub use plan::{build_merkle_tree, get_dispatch_linear, WebGpuHelper};

/// Supported Hash functions
#[derive(Copy, Clone, PartialEq)]
pub enum HashFn {
    Rpo256,
    Rpx256,
}

impl core::fmt::Display for HashFn {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HashFn::Rpo256 => write!(f, "rpo_256"),
            HashFn::Rpx256 => write!(f, "rpx_256"),
        }
    }
}
