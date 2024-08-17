extern crate alloc;

use alloc::{boxed::Box, vec::Vec};

use once_cell::race::OnceBox;
use wgpu::util::DeviceExt;
use winter_math::fields::f64::BaseElement;

use crate::HashFn;

const LIBRARY_DATA: &'static str = include_str!("shaders/hash_shaders/rpx_shader.wgsl");

pub fn log2(n: usize) -> u32 {
    assert!(n.is_power_of_two(), "n must be a power of two");
    n.trailing_zeros()
}

#[derive(Debug)]
pub struct WebGpuHelper {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub module: wgpu::ShaderModule,
}
const DISPATCH_MAX_PER_DIM: u64 = 32768u64;

pub fn get_dispatch_linear(size: u64) -> (u32, u32, u32) {
    if size <= DISPATCH_MAX_PER_DIM {
        return (size as u32, 1, 1);
    } else if size <= DISPATCH_MAX_PER_DIM * DISPATCH_MAX_PER_DIM {
        assert_eq!(size % DISPATCH_MAX_PER_DIM, 0);
        return (DISPATCH_MAX_PER_DIM as u32, (size / DISPATCH_MAX_PER_DIM) as u32, 1);
    } else if size <= DISPATCH_MAX_PER_DIM * DISPATCH_MAX_PER_DIM * DISPATCH_MAX_PER_DIM {
        assert_eq!(size % (DISPATCH_MAX_PER_DIM * DISPATCH_MAX_PER_DIM), 0);
        return (
            DISPATCH_MAX_PER_DIM as u32,
            DISPATCH_MAX_PER_DIM as u32,
            (size / (DISPATCH_MAX_PER_DIM * DISPATCH_MAX_PER_DIM)) as u32,
        );
    } else {
        panic!("size too large for dispatch");
    }
}

impl WebGpuHelper {
    pub async fn new() -> Option<Self> {
        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await?;

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .ok()?;

        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rpx_shader"),
            source: wgpu::ShaderSource::Wgsl(LIBRARY_DATA.into()),
        });

        Some(Self { device, queue, module: cs_module })
    }
}

// TODO: unsafe
unsafe impl Send for WebGpuHelper {}
unsafe impl Sync for WebGpuHelper {}

static WGPU_HELPER: OnceBox<WebGpuHelper> = OnceBox::new();

pub async fn get_wgpu_helper() -> &'static WebGpuHelper {
    let wgpu_helper = WGPU_HELPER.get();

    if wgpu_helper.is_none() {
        let helper = WebGpuHelper::new().await.unwrap();
        WGPU_HELPER.set(Box::new(helper)).unwrap();
        WGPU_HELPER.get().unwrap()
    } else {
        wgpu_helper.unwrap()
    }
}

pub async fn build_merkle_tree(
    helper: &WebGpuHelper,
    leaves: &[[BaseElement; 4]],
    hash_fn: HashFn,
) -> Option<Vec<[BaseElement; 4]>> {
    let rpx = hash_fn == HashFn::Rpx256;
    let height = log2(leaves.len());
    let digest_size = 4 * core::mem::size_of::<BaseElement>() as usize;
    let tree_nodes_size = digest_size * leaves.len();
    let staging_buffer = helper.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: tree_nodes_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let leaf_digests_buffer = helper.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("leaf_digests"),
        contents: unsafe {
            core::slice::from_raw_parts(leaves.as_ptr() as *const u8, leaves.len() * digest_size)
        },
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let nodes_buffer = helper.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Nodes Buffer"),
        size: tree_nodes_size as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let rpo_hash_leaves_pipeline =
        helper.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &helper.module,
            entry_point: if rpx { "rpx_hash_leaves" } else { "rpo_hash_leaves" },
            compilation_options: Default::default(),
            cache: None,
        });
    let mut node_count_value: [u32; 1] = [(leaves.len() / 2usize) as u32];

    let node_count_buf = helper.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("node_count"),
        contents: bytemuck::cast_slice(&node_count_value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let mut encoder = helper
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = rpo_hash_leaves_pipeline.get_bind_group_layout(0);
    let bind_group = helper.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: nodes_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: node_count_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: leaf_digests_buffer.as_entire_binding(),
            },
        ],
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&rpo_hash_leaves_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        #[cfg(debug_assertions)]
        cpass.insert_debug_marker("compute hash leaves");

        let dispatch_dims = get_dispatch_linear(node_count_value[0] as u64);

        cpass.dispatch_workgroups(dispatch_dims.0, dispatch_dims.1, dispatch_dims.2);
        // Number of cells to run, the (x,y,z) size of item being processed
    }

    // start merkle hashing
    let rpo_hash_level_pipeline =
        helper.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &helper.module,
            entry_point: if rpx { "rpx_hash_level" } else { "rpo_hash_level" },
            compilation_options: Default::default(),
            cache: None,
        });

    for _ in 1..height {
        node_count_value[0] >>= 1;
        let node_count_buf = helper.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("node_count"),
            contents: bytemuck::cast_slice(&node_count_value),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group_layout = rpo_hash_level_pipeline.get_bind_group_layout(0);
        let bind_group = helper.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: nodes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: node_count_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&rpo_hash_level_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            #[cfg(debug_assertions)]
            cpass.insert_debug_marker("compute hash leaves");

            let dispatch_dims = get_dispatch_linear(node_count_value[0] as u64);

            cpass.dispatch_workgroups(dispatch_dims.0, dispatch_dims.1, dispatch_dims.2);
            // Number of cells to run, the (x,y,z) size of item being processed
        }
    }

    // end merkle hashing

    // get result

    encoder.copy_buffer_to_buffer(&nodes_buffer, 0, &staging_buffer, 0, tree_nodes_size as u64);

    helper.queue.submit(Some(encoder.finish()));
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.

    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    helper.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // Awaits until `buffer_future` can be read from
    if let Ok(Ok(())) = receiver.recv_async().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        //let result:Vec<u64> = bytemuck::cast_slice(&data).to_vec();
        let result = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *mut [BaseElement; 4], leaves.len())
        };

        /*
        let result = unsafe {
          Vec::from_raw_parts(data.as_ptr() as *mut [Felt; 4], n, n)
        };*/

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory

        // Returns data from buffer
        Some(result.to_vec())
    } else {
        panic!("failed to merkle hasher on web gpu!")
    }
}

#[cfg(test)]
mod tests {
    use miden_crypto::{
        hash::{
            rpo::{Rpo256, RpoDigest},
            rpx::{Rpx256, RpxDigest},
        },
        Felt,
    };
    #[cfg(target_family = "wasm")]
    use wasm_bindgen::prelude::*;
    #[cfg(target_family = "wasm")]
    use wasm_bindgen_test::*;
    use winter_crypto::MerkleTree;

    use super::*;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test_configure!(run_in_browser);

    #[cfg(target_family = "wasm")]
    extern crate wee_alloc;
    #[cfg(target_family = "wasm")]
    #[global_allocator]
    static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

    fn webgpu_test_merkle_tree(num_leaves: usize, hash_fn: HashFn) {
        let helper = pollster::block_on(WebGpuHelper::new()).unwrap();
        let leaves: Vec<[BaseElement; 4]> =
            (0..num_leaves).map(|i| [Felt::new(i as u64); 4]).collect();
        let gpu_result = pollster::block_on(build_merkle_tree(&helper, &leaves, hash_fn)).unwrap();
        let cpu_result = match hash_fn {
            HashFn::Rpo256 => MerkleTree::<Rpo256>::new(
                leaves.into_iter().map(|leaf| RpoDigest::new(leaf)).collect::<Vec<_>>(),
            )
            .unwrap()
            .root()
            .to_vec(),
            HashFn::Rpx256 => MerkleTree::<Rpx256>::new(
                leaves.into_iter().map(|leaf| RpxDigest::new(leaf)).collect::<Vec<_>>(),
            )
            .unwrap()
            .root()
            .to_vec(),
        };
        assert_eq!(cpu_result, gpu_result[1]);
    }

    #[cfg(target_family = "wasm")]
    #[wasm_bindgen]
    pub async fn wasm_test_merkle_tree(num_leaves: usize) {
        console_error_panic_hook::set_once();
        let helper = WebGpuHelper::new().await.unwrap();
        let leaves: Vec<[BaseElement; 4]> =
            (0..num_leaves).map(|i| [Felt::new(i as u64); 4]).collect();
        let gpu_result = build_merkle_tree(&helper, &leaves, HashFn::Rpo256).await.unwrap();
        let cpu_result = MerkleTree::<Rpo256>::new(
            leaves.into_iter().map(|leaf| RpoDigest::new(leaf)).collect::<Vec<_>>(),
        )
        .unwrap()
        .root()
        .to_vec();
        assert_eq!(cpu_result, gpu_result[1]);
    }

    #[test]
    fn webgpu_rpo_small_merkle_tree() {
        webgpu_test_merkle_tree(4, HashFn::Rpo256);
    }

    #[test]
    fn webgpu_rpo_medium_merkle_tree() {
        webgpu_test_merkle_tree(1 << 10, HashFn::Rpo256);
    }

    #[test]
    fn webgpu_rpo_large_merkle_tree() {
        webgpu_test_merkle_tree(1 << 15, HashFn::Rpo256);
    }

    #[test]
    fn webgpu_rpx_small_merkle_tree() {
        webgpu_test_merkle_tree(4, HashFn::Rpx256);
    }

    #[test]
    fn webgpu_rpx_medium_merkle_tree() {
        webgpu_test_merkle_tree(1 << 10, HashFn::Rpx256);
    }

    #[test]
    fn webgpu_rpx_large_merkle_tree() {
        webgpu_test_merkle_tree(1 << 15, HashFn::Rpx256);
    }

    #[cfg(target_family = "wasm")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn wasm_rpo_medium_merkle_tree() {
        wasm_bindgen_futures::spawn_local(wasm_test_merkle_tree(1 << 10));
    }
}
