extern crate alloc;

use alloc::{vec, vec::Vec};

use wgpu::util::DeviceExt;
use winter_math::fields::f64::BaseElement;

use super::plan::{get_dispatch_linear, WebGpuHelper};
use crate::HashFn;

pub struct RowHasher {
    row_hash_state_buffer: wgpu::Buffer,
    digests_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    encoder: wgpu::CommandEncoder,
    pipeline: wgpu::ComputePipeline,
    pad_pipeline: wgpu::ComputePipeline,
    n: usize,
    pad_columns: bool,
}

impl RowHasher {
    pub fn new(helper: &WebGpuHelper, n: usize, pad_columns: bool, hash_fn: HashFn) -> Self {
        let rpx = hash_fn == HashFn::Rpx256;
        let device = &helper.device;
        let row_state_size = (4 * n * core::mem::size_of::<BaseElement>()) as wgpu::BufferAddress;
        let digests_size = (4 * n * core::mem::size_of::<BaseElement>()) as wgpu::BufferAddress;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: digests_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let row_hash_state_buffer = helper.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Row State Buffer"),
            size: row_state_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let digests_buffer = helper.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Digests Buffer"),
            size: digests_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        helper.queue.write_buffer(
            &row_hash_state_buffer,
            0,
            bytemuck::cast_slice(&vec![[0u64; 4]; n]),
        );
        let encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &helper.module,
            entry_point: if rpx { "rpx_absorb_rows" } else { "rpo_absorb_rows" },
            compilation_options: Default::default(),
            cache: None,
        });

        let compute_pipeline_pad =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &helper.module,
                entry_point: if rpx {
                    "rpx_absorb_rows_pad"
                } else {
                    "rpo_absorb_rows_pad"
                },
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            staging_buffer,
            encoder,
            pipeline: compute_pipeline,
            pad_pipeline: compute_pipeline_pad,
            digests_buffer,
            row_hash_state_buffer,
            pad_columns,
            n,
        }
    }
    pub fn update(&mut self, helper: &WebGpuHelper, rows: &[[BaseElement; 8]]) {
        let rows_ptr = unsafe {
            core::slice::from_raw_parts(
                rows.as_ptr() as *mut u8,
                rows.len() * core::mem::size_of::<BaseElement>() * 8,
            )
        };
        let row_input_buffer =
            helper.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rows"),
                contents: rows_ptr,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        let pipeline = if self.pad_columns {
            self.pad_columns = false;
            &self.pad_pipeline
        } else {
            &self.pipeline
        };

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let bind_group = helper.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.digests_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: row_input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.row_hash_state_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            #[cfg(debug_assertions)]
            cpass.insert_debug_marker("compute row absorb");

            let dispatch_dims = get_dispatch_linear(rows.len() as u64);

            cpass.dispatch_workgroups(dispatch_dims.0, dispatch_dims.1, dispatch_dims.2);
        }
    }
    pub async fn finish(self, helper: &WebGpuHelper) -> Option<Vec<[BaseElement; 4]>> {
        let staging_buffer = self.staging_buffer;
        let results_buffer = self.digests_buffer;
        let mut encoder = self.encoder;
        let queue = &helper.queue;
        let n = self.n;
        encoder.copy_buffer_to_buffer(
            &results_buffer,
            0,
            &staging_buffer,
            0,
            (n * 4 * core::mem::size_of::<BaseElement>()) as u64,
        );

        queue.submit(Some(encoder.finish()));
        let buffer_slice = staging_buffer.slice(..);
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
            let result =
                unsafe { core::slice::from_raw_parts(data.as_ptr() as *mut [BaseElement; 4], n) };

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
            panic!("failed to run compute on gpu!")
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use miden_crypto::{
        hash::{rpo::Rpo256, rpx::Rpx256},
        Felt,
    };
    #[cfg(target_family = "wasm")]
    use wasm_bindgen::prelude::*;
    #[cfg(target_family = "wasm")]
    use wasm_bindgen_test::*;

    use super::*;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test_configure!(run_in_browser);

    fn webgpu_test_row_hash(num_rows: usize, hash_fn: HashFn) {
        let helper = pollster::block_on(WebGpuHelper::new()).unwrap();
        let mut rpo = RowHasher::new(&helper, num_rows, false, hash_fn);
        let rows: Vec<[BaseElement; 8]> =
            vec![core::array::from_fn(|i| Felt::new(i as u64)); num_rows];
        rpo.update(&helper, &rows);
        let gpu_results = pollster::block_on(rpo.finish(&helper)).unwrap();
        assert_eq!(gpu_results.len(), rows.len());
        gpu_results.iter().zip(rows.iter()).for_each(|(gpu_result, row)| {
            if hash_fn == HashFn::Rpo256 {
                let cpu_result = Rpo256::hash_elements(row.as_slice());
                assert_eq!(gpu_result, cpu_result.as_elements());
            } else {
                let cpu_result = Rpx256::hash_elements(row.as_slice());
                assert_eq!(gpu_result, cpu_result.as_elements());
            }
        });
    }

    #[cfg(target_family = "wasm")]
    #[wasm_bindgen]
    pub async fn wasm_test_row_hash(num_rows: usize) {
        console_error_panic_hook::set_once();
        let helper = WebGpuHelper::new().await.unwrap();
        let mut rpo = RowHasher::new(&helper, num_rows, false, HashFn::Rpo256);
        let rows: Vec<[BaseElement; 8]> =
            vec![core::array::from_fn(|i| Felt::new(i as u64)); num_rows];
        rpo.update(&helper, &rows);
        let gpu_results = rpo.finish(&helper).await.unwrap();
        assert_eq!(gpu_results.len(), rows.len());
        gpu_results.iter().zip(rows.iter()).for_each(|(gpu_result, row)| {
            let cpu_result = Rpo256::hash_elements(row.as_slice());
            assert_eq!(gpu_result, cpu_result.as_elements());
        });
    }

    #[test]
    fn webgpu_rpo_from_single_row() {
        webgpu_test_row_hash(1, HashFn::Rpo256);
    }

    #[test]
    fn webgpu_rpo_from_medium_sized_rows() {
        webgpu_test_row_hash(1 << 15, HashFn::Rpo256);
    }

    #[test]
    fn webgpu_rpo_from_large_sized_rows() {
        webgpu_test_row_hash(1 << 20, HashFn::Rpo256);
    }

    #[test]
    fn webgpu_rpx_from_single_row() {
        webgpu_test_row_hash(1, HashFn::Rpx256);
    }

    #[test]
    fn webgpu_rpx_from_medium_sized_rows() {
        webgpu_test_row_hash(1 << 15, HashFn::Rpx256);
    }

    #[test]
    fn webgpu_rpx_from_large_sized_rows() {
        webgpu_test_row_hash(1 << 20, HashFn::Rpx256);
    }

    #[cfg(target_family = "wasm")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn wasm_rpo_from_medium_sized_rows() {
        wasm_bindgen_futures::spawn_local(wasm_test_row_hash(1 << 5));
    }
}
