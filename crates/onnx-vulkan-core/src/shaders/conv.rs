//! Shared shaders and dispatch layouts for floating point `Conv`.
//!
//! Two variants with identical bindings and push constants, chosen based on
//! `group`:
//!
//! - `CONV_F32_GEMM` for `group == 1`: implicit tiled GEMM, how GPUs
//!   compute convolutions (each value loaded into shared memory
//!   serves 16 threads instead of one).
//! - `CONV_F32_DIRECT` for grouped/depthwise convolutions, which are not
//!   a single GEMM: each output channel views only its own
//!   group of input channels, so row blocks cannot share loads.
//!   of the same im2col column.

pub const BINDINGS: u32 = 4;
/// 19 u32 fields in the push constant struct. The last is `split`, read only
/// by [`blocked_splitk`]; the single-pass kernels ignore it, so one layout
/// serves every variant.
pub const PUSH_BYTES: u32 = 76;
/// Threads used by the direct kernel's one-dimensional workgroup.
pub const DIRECT_WORKGROUP_SIZE: u32 = 256;
pub const TILE_SIZE: u32 = 16;
/// Number of reduction-axis values staged per blocked iteration.
pub const BLOCKED_K_STEP: u32 = 16;
/// Outputs computed per axis by one blocked-kernel invocation.
pub const BLOCKED_MICRO_TILE_SIZE: u32 = 4;
/// Output tile of [`blocked`], which is otherwise a drop-in for
/// [`implicit_gemm`] — same bindings, same push constants, only the grid
/// divisor changes.
pub const BLOCKED_TILE_SIZE: u32 = 64;

/// Fully specified floating-point convolution implementation.
///
/// These values describe the parameters compiled into the currently shipped
/// shaders. Keeping them in the tactic identity makes the production route a
/// reusable control for future candidate generation instead of an implicit
/// tuple assembled in the interpreter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tactic {
    Direct {
        workgroup_size: u32,
    },
    ImplicitGemm {
        output_tile: u32,
        k_step: u32,
    },
    Blocked {
        output_tile: u32,
        k_step: u32,
        micro_tile: u32,
    },
    SplitK {
        output_tile: u32,
        k_step: u32,
        micro_tile: u32,
        split: u32,
    },
}

/// Normalized dimensions needed to decide whether a tactic can execute.
///
/// `kdepth` is per group (`C_in / group * KH * KW`) and `total` is the number
/// of output elements. Wider arithmetic is deliberate: malformed or oversized
/// candidate metadata is rejected before it can be narrowed into Vulkan's
/// `u32` dispatch dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TacticGeometry {
    pub batch: u32,
    pub group: u32,
    pub pixels: u32,
    pub c_out: u32,
    pub kdepth: u32,
    pub total: u64,
}

/// Device limits plus the caller's per-trial scratch policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TacticLimits {
    pub max_workgroup_count: [u32; 3],
    pub max_workgroup_invocations: u32,
    pub max_workgroup_size: [u32; 3],
    pub max_shared_memory_bytes: u32,
    pub max_storage_buffer_bytes: u64,
    pub max_scratch_bytes: u64,
}

impl TacticLimits {
    pub const fn from_vulkan(limits: &vk_compute::ComputeLimits, max_scratch_bytes: u64) -> Self {
        Self {
            max_workgroup_count: limits.max_workgroup_count,
            max_workgroup_invocations: limits.max_workgroup_invocations,
            max_workgroup_size: limits.max_workgroup_size,
            max_shared_memory_bytes: limits.max_shared_memory_bytes,
            max_storage_buffer_bytes: limits.max_storage_buffer_bytes,
            max_scratch_bytes,
        }
    }
}

/// Stable reason why a Conv tactic is impossible for a geometry/device pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TacticRejection {
    EmptyGeometry,
    InconsistentOutputElements,
    InvalidGroup,
    GroupedGemm,
    InvalidParameters,
    EmptySplit,
    WorkgroupLimit,
    SharedMemoryLimit,
    GridLimit,
    StorageBufferLimit,
    ScratchBudget,
    ArithmeticOverflow,
}

pub const CANDIDATE_OUTPUT_TILES: &[u32] = &[8, 16, 32, 64, 128];
pub const CANDIDATE_K_STEPS: &[u32] = &[8, 16, 32, 64];
pub const CANDIDATE_MICRO_TILES: &[u32] = &[1, 2, 4, 8];
pub const CANDIDATE_DIRECT_WORKGROUPS: &[u32] = &[64, 128, 256, 512, 1024];
pub const CANDIDATE_SPLITS: std::ops::RangeInclusive<u32> = 2..=32;

/// Complete metadata space before geometry/device viability filtering.
///
/// Grouped/depthwise convolutions can only vary the direct workgroup. A
/// `group == 1` geometry explores the implicit tile plus blocked single-pass
/// and split-K forms. Every split in the bounded range is present rather than
/// only powers of two, so the shipped control split is never omitted.
pub fn candidate_space(group: u32) -> Vec<Tactic> {
    if group != 1 {
        return CANDIDATE_DIRECT_WORKGROUPS
            .iter()
            .map(|&workgroup_size| Tactic::Direct { workgroup_size })
            .collect();
    }

    let mut candidates = Vec::new();
    for &output_tile in CANDIDATE_OUTPUT_TILES {
        for &k_step in CANDIDATE_K_STEPS {
            candidates.push(Tactic::ImplicitGemm {
                output_tile,
                k_step,
            });
            for &micro_tile in CANDIDATE_MICRO_TILES {
                candidates.push(Tactic::Blocked {
                    output_tile,
                    k_step,
                    micro_tile,
                });
                for split in CANDIDATE_SPLITS {
                    candidates.push(Tactic::SplitK {
                        output_tile,
                        k_step,
                        micro_tile,
                        split,
                    });
                }
            }
        }
    }
    candidates
}

pub const REDUCED_OUTPUT_TILES: &[u32] = &[8, 16, 32, 64, 128];
pub const REDUCED_K_STEPS: &[u32] = &[8, 16, 32, 64];
pub const REDUCED_SPLITS: &[u32] = &[2, 4, 8, 16, 32];

/// Executable first-pass space used before considering a sampler.
///
/// It keeps the shipped four-output register micro-tile and varies the axes
/// whose code generator is already auditable against the committed kernels.
/// The full metadata space remains the authority for reporting and future
/// expansion; this deliberately smaller set bounds compilation and timing.
pub fn reduced_exhaustive_space() -> Vec<Tactic> {
    let mut candidates = REDUCED_OUTPUT_TILES
        .iter()
        .map(|&output_tile| Tactic::ImplicitGemm {
            output_tile,
            k_step: output_tile,
        })
        .collect::<Vec<_>>();
    for &output_tile in REDUCED_OUTPUT_TILES {
        for &k_step in REDUCED_K_STEPS {
            candidates.push(Tactic::Blocked {
                output_tile,
                k_step,
                micro_tile: BLOCKED_MICRO_TILE_SIZE,
            });
            for &split in REDUCED_SPLITS {
                candidates.push(Tactic::SplitK {
                    output_tile,
                    k_step,
                    micro_tile: BLOCKED_MICRO_TILE_SIZE,
                    split,
                });
            }
        }
    }
    candidates
}

/// Census-wide first search: 21 points plus the shipped control per geometry.
///
/// Earlier probes already ruled out broad K-step changes at batch one, so the
/// first exhaustive pass keeps `KSTEP=16`, varies output occupancy, and crosses
/// the same three tiles with the bounded power-of-two split ladder.
pub fn focused_search_space() -> Vec<Tactic> {
    let mut candidates = [8, 16, 32]
        .into_iter()
        .map(|output_tile| Tactic::ImplicitGemm {
            output_tile,
            k_step: output_tile,
        })
        .collect::<Vec<_>>();
    for output_tile in [32, 64, 128] {
        candidates.push(Tactic::Blocked {
            output_tile,
            k_step: BLOCKED_K_STEP,
            micro_tile: BLOCKED_MICRO_TILE_SIZE,
        });
        for &split in REDUCED_SPLITS {
            candidates.push(Tactic::SplitK {
                output_tile,
                k_step: BLOCKED_K_STEP,
                micro_tile: BLOCKED_MICRO_TILE_SIZE,
                split,
            });
        }
    }
    candidates
}

/// Generates an executable shader for the reduced candidate family.
///
/// Returning `None` is distinct from static viability: the full metadata space
/// intentionally describes future axes that this first generator does not yet
/// implement. The shipped parameter points reproduce the committed sources
/// byte-for-byte, which is covered by tests.
pub fn generated_source(tactic: Tactic) -> Option<String> {
    match tactic {
        Tactic::Direct { workgroup_size } if workgroup_size > 0 => Some(direct().replace(
            "@compute @workgroup_size(256)",
            &format!("@compute @workgroup_size({workgroup_size})"),
        )),
        Tactic::ImplicitGemm {
            output_tile,
            k_step,
        } if output_tile > 0 && output_tile == k_step => {
            let staged = output_tile.checked_mul(output_tile)?;
            Some(
                implicit_gemm()
                    .replace(
                        "const TILE = 16u;",
                        &format!("const TILE = {output_tile}u;"),
                    )
                    .replace(
                        "var<workgroup> w_tile: array<f32, 256>;",
                        &format!("var<workgroup> w_tile: array<f32, {staged}>;"),
                    )
                    .replace(
                        "var<workgroup> x_tile: array<f32, 256>;",
                        &format!("var<workgroup> x_tile: array<f32, {staged}>;"),
                    )
                    .replace(
                        "@compute @workgroup_size(16, 16)",
                        &format!("@compute @workgroup_size({output_tile}, {output_tile})"),
                    ),
            )
        }
        Tactic::Blocked {
            output_tile,
            k_step,
            micro_tile: BLOCKED_MICRO_TILE_SIZE,
        } => generated_blocked(output_tile, k_step, false),
        Tactic::SplitK {
            output_tile,
            k_step,
            micro_tile: BLOCKED_MICRO_TILE_SIZE,
            ..
        } => generated_blocked(output_tile, k_step, true),
        _ => None,
    }
}

fn generated_blocked(output_tile: u32, k_step: u32, split: bool) -> Option<String> {
    if output_tile == 0 || k_step == 0 || !output_tile.is_multiple_of(BLOCKED_MICRO_TILE_SIZE) {
        return None;
    }
    let side = output_tile / BLOCKED_MICRO_TILE_SIZE;
    let invocations = side.checked_mul(side)?;
    let staged = output_tile.checked_mul(k_step)?;
    if !staged.is_multiple_of(invocations) {
        return None;
    }
    let loads = staged / invocations;
    Some(
        blocked_body(split)
            .replace(
                "const TILE = 64u;",
                &format!("const TILE = {output_tile}u;"),
            )
            .replace("const KSTEP = 16u;", &format!("const KSTEP = {k_step}u;"))
            .replace(
                "var<workgroup> w_tile: array<f32, 1024>;",
                &format!("var<workgroup> w_tile: array<f32, {staged}>;"),
            )
            .replace(
                "var<workgroup> x_tile: array<f32, 1024>;",
                &format!("var<workgroup> x_tile: array<f32, {staged}>;"),
            )
            .replace(
                "@compute @workgroup_size(16, 16)",
                &format!("@compute @workgroup_size({side}, {side})"),
            )
            .replace(
                "let tid = lid.y * 16u + lid.x;",
                &format!("let tid = lid.y * {side}u + lid.x;"),
            )
            .replace(
                "for (var s = 0u; s < 4u; s = s + 1u)",
                &format!("for (var s = 0u; s < {loads}u; s = s + 1u)"),
            )
            .replace(
                "let l = tid + s * 256u;",
                &format!("let l = tid + s * {invocations}u;"),
            ),
    )
}

impl Tactic {
    /// Persistent ID and parameters accepted by the runtime artifact boundary.
    pub fn persistent_parts(self) -> (crate::tuning::TacticId, crate::tuning::TacticParameters) {
        use std::collections::BTreeMap;
        let (variant, parameters) = match self {
            Self::Direct { workgroup_size } => (
                "direct",
                BTreeMap::from([("workgroup_size".into(), i64::from(workgroup_size))]),
            ),
            Self::ImplicitGemm {
                output_tile,
                k_step,
            } => (
                "implicit-gemm",
                BTreeMap::from([
                    ("k_step".into(), i64::from(k_step)),
                    ("output_tile".into(), i64::from(output_tile)),
                ]),
            ),
            Self::Blocked {
                output_tile,
                k_step,
                micro_tile,
            } => (
                "blocked",
                BTreeMap::from([
                    ("k_step".into(), i64::from(k_step)),
                    ("micro_tile".into(), i64::from(micro_tile)),
                    ("output_tile".into(), i64::from(output_tile)),
                ]),
            ),
            Self::SplitK {
                output_tile,
                k_step,
                micro_tile,
                split,
            } => (
                "split-k",
                BTreeMap::from([
                    ("k_step".into(), i64::from(k_step)),
                    ("micro_tile".into(), i64::from(micro_tile)),
                    ("output_tile".into(), i64::from(output_tile)),
                    ("split".into(), i64::from(split)),
                ]),
            ),
        };
        (
            crate::tuning::TacticId::new("conv-f32".into(), variant.into()),
            parameters,
        )
    }

    /// Parses only the finite tactic contract owned by this kernel family.
    pub fn from_persistent(
        id: &crate::tuning::TacticId,
        parameters: &crate::tuning::TacticParameters,
    ) -> Option<Self> {
        if id.family() != "conv-f32" {
            return None;
        }
        let parameter = |name: &str| u32::try_from(*parameters.get(name)?).ok();
        let tactic = match id.variant() {
            "direct" if parameters.len() == 1 => Self::Direct {
                workgroup_size: parameter("workgroup_size")?,
            },
            "implicit-gemm" if parameters.len() == 2 => Self::ImplicitGemm {
                output_tile: parameter("output_tile")?,
                k_step: parameter("k_step")?,
            },
            "blocked" if parameters.len() == 3 => Self::Blocked {
                output_tile: parameter("output_tile")?,
                k_step: parameter("k_step")?,
                micro_tile: parameter("micro_tile")?,
            },
            "split-k" if parameters.len() == 4 => Self::SplitK {
                output_tile: parameter("output_tile")?,
                k_step: parameter("k_step")?,
                micro_tile: parameter("micro_tile")?,
                split: parameter("split")?,
            },
            _ => return None,
        };
        candidate_space(if matches!(tactic, Self::Direct { .. }) {
            2
        } else {
            1
        })
        .contains(&tactic)
        .then_some(tactic)
    }

    /// Number of partial images produced by this tactic.
    pub const fn split(self) -> u32 {
        match self {
            Self::SplitK { split, .. } => split,
            _ => 1,
        }
    }

    /// Dispatch grid for the primary convolution pass.
    pub fn dispatch_grid(self, total: u32, pixels: u32, c_out: u32, batch: u32) -> [u32; 3] {
        match self {
            Self::Direct { workgroup_size } => [total.div_ceil(workgroup_size), 1, 1],
            Self::ImplicitGemm { output_tile, .. }
            | Self::Blocked { output_tile, .. }
            | Self::SplitK { output_tile, .. } => [
                pixels.div_ceil(output_tile),
                c_out.div_ceil(output_tile),
                batch.max(1) * self.split(),
            ],
        }
    }

    /// Rejects a tactic that cannot execute on this geometry and device.
    ///
    /// Output and reduction padding are supported by every GEMM-family shader,
    /// so non-divisible `pixels`, `c_out`, and `kdepth` use ceiling division.
    /// Divisibility is required only between compile-time tile parameters where
    /// the shader distributes staged values evenly across its workgroup.
    pub fn viability(
        self,
        geometry: TacticGeometry,
        limits: TacticLimits,
    ) -> Result<(), TacticRejection> {
        let TacticGeometry {
            batch,
            group,
            pixels,
            c_out,
            kdepth,
            total,
        } = geometry;
        if batch == 0 || pixels == 0 || c_out == 0 || kdepth == 0 || total == 0 {
            return Err(TacticRejection::EmptyGeometry);
        }
        let expected_total = u64::from(batch)
            .checked_mul(u64::from(c_out))
            .and_then(|value| value.checked_mul(u64::from(pixels)))
            .ok_or(TacticRejection::ArithmeticOverflow)?;
        if total != expected_total || total > u64::from(u32::MAX) {
            return Err(TacticRejection::InconsistentOutputElements);
        }
        if group == 0 || group > c_out || !c_out.is_multiple_of(group) {
            return Err(TacticRejection::InvalidGroup);
        }
        if group != 1 && !matches!(self, Self::Direct { .. }) {
            return Err(TacticRejection::GroupedGemm);
        }

        let (workgroup, shared_bytes, output_tile) = self.resources(kdepth)?;
        let invocations = workgroup
            .iter()
            .try_fold(1_u64, |product, &dimension| {
                product.checked_mul(u64::from(dimension))
            })
            .ok_or(TacticRejection::ArithmeticOverflow)?;
        if invocations > u64::from(limits.max_workgroup_invocations)
            || workgroup
                .iter()
                .zip(limits.max_workgroup_size)
                .any(|(&actual, maximum)| actual > maximum)
        {
            return Err(TacticRejection::WorkgroupLimit);
        }
        if shared_bytes > u64::from(limits.max_shared_memory_bytes) {
            return Err(TacticRejection::SharedMemoryLimit);
        }

        let split = u64::from(self.split());
        let grid = match output_tile {
            Some(tile) => [
                u64::from(pixels.div_ceil(tile)),
                u64::from(c_out.div_ceil(tile)),
                u64::from(batch)
                    .checked_mul(split)
                    .ok_or(TacticRejection::ArithmeticOverflow)?,
            ],
            None => [total.div_ceil(u64::from(workgroup[0])), 1, 1],
        };
        if grid
            .iter()
            .zip(limits.max_workgroup_count)
            .any(|(&actual, maximum)| actual > u64::from(maximum))
        {
            return Err(TacticRejection::GridLimit);
        }

        let output_bytes = total
            .checked_mul(size_of::<f32>() as u64)
            .ok_or(TacticRejection::ArithmeticOverflow)?;
        if output_bytes > limits.max_storage_buffer_bytes {
            return Err(TacticRejection::StorageBufferLimit);
        }

        if split > 1 {
            let scratch_bytes = total
                .checked_mul(split)
                .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
                .ok_or(TacticRejection::ArithmeticOverflow)?;
            if scratch_bytes > limits.max_storage_buffer_bytes {
                return Err(TacticRejection::StorageBufferLimit);
            }
            if scratch_bytes > limits.max_scratch_bytes {
                return Err(TacticRejection::ScratchBudget);
            }
        }
        Ok(())
    }

    pub fn is_viable(self, geometry: TacticGeometry, limits: TacticLimits) -> bool {
        self.viability(geometry, limits).is_ok()
    }

    fn resources(self, kdepth: u32) -> Result<([u32; 3], u64, Option<u32>), TacticRejection> {
        match self {
            Self::Direct { workgroup_size } => {
                if workgroup_size == 0 {
                    return Err(TacticRejection::InvalidParameters);
                }
                Ok(([workgroup_size, 1, 1], 0, None))
            }
            Self::ImplicitGemm {
                output_tile,
                k_step,
            } => {
                if output_tile == 0 || k_step == 0 {
                    return Err(TacticRejection::InvalidParameters);
                }
                let invocations = output_tile
                    .checked_mul(output_tile)
                    .ok_or(TacticRejection::ArithmeticOverflow)?;
                let staged = output_tile
                    .checked_mul(k_step)
                    .ok_or(TacticRejection::ArithmeticOverflow)?;
                if !staged.is_multiple_of(invocations) {
                    return Err(TacticRejection::InvalidParameters);
                }
                Ok((
                    [output_tile, output_tile, 1],
                    u64::from(staged) * 2 * size_of::<f32>() as u64,
                    Some(output_tile),
                ))
            }
            Self::Blocked {
                output_tile,
                k_step,
                micro_tile,
            }
            | Self::SplitK {
                output_tile,
                k_step,
                micro_tile,
                ..
            } => {
                if output_tile == 0
                    || k_step == 0
                    || micro_tile == 0
                    || !output_tile.is_multiple_of(micro_tile)
                {
                    return Err(TacticRejection::InvalidParameters);
                }
                let side = output_tile / micro_tile;
                let invocations = side
                    .checked_mul(side)
                    .ok_or(TacticRejection::ArithmeticOverflow)?;
                let staged = output_tile
                    .checked_mul(k_step)
                    .ok_or(TacticRejection::ArithmeticOverflow)?;
                if !staged.is_multiple_of(invocations) {
                    return Err(TacticRejection::InvalidParameters);
                }
                if let Self::SplitK { split, .. } = self {
                    if split <= 1 || split > kdepth.div_ceil(k_step) {
                        return Err(TacticRejection::EmptySplit);
                    }
                }
                Ok((
                    [side, side, 1],
                    u64::from(staged) * 2 * size_of::<f32>() as u64,
                    Some(output_tile),
                ))
            }
        }
    }
}

/// Fingerprint of the complete executable Conv-f32 tactic family.
pub fn implementation_fingerprint() -> crate::tuning::ImplementationFingerprint {
    use crate::tuning::{
        ImplementationFingerprint, ImplementationFingerprintInputs, NamedShaderSource,
    };
    use std::collections::BTreeMap;
    static FINGERPRINT: std::sync::OnceLock<ImplementationFingerprint> = std::sync::OnceLock::new();
    *FINGERPRINT.get_or_init(|| {
        let blocked = blocked();
        let direct = direct();
        let implicit_gemm = implicit_gemm();
        let split_k = blocked_splitk();
        let sources = [
            NamedShaderSource {
                name: "blocked",
                contents: blocked.as_bytes(),
            },
            NamedShaderSource {
                name: "direct",
                contents: direct.as_bytes(),
            },
            NamedShaderSource {
                name: "implicit_gemm",
                contents: implicit_gemm.as_bytes(),
            },
            NamedShaderSource {
                name: "split_k",
                contents: split_k.as_bytes(),
            },
            NamedShaderSource {
                name: "split_reduce",
                contents: SPLIT_REDUCE.as_bytes(),
            },
        ];
        ImplementationFingerprint::compute(&ImplementationFingerprintInputs {
            schema_version: 1,
            shader_sources: &sources,
            specialization_constants: &BTreeMap::new(),
            generator_logic: include_bytes!("conv.rs"),
            dispatch_logic: b"interp::conv_f32:v1",
            compiler_settings: b"naga=30;wgsl-in;spv-out;entry=main",
            runtime_abi: b"bindings=4;push=68;split-reduce-bindings=3",
        })
        .expect("the committed Conv implementation fingerprint is complete")
    })
}

/// How many workgroups [`blocked`] would launch for this output.
fn workgroups(pixels: usize, c_out: usize) -> usize {
    pixels.div_ceil(BLOCKED_TILE_SIZE as usize) * c_out.div_ceil(BLOCKED_TILE_SIZE as usize)
}

/// Fewest workgroups at which the 64×64 tile still pays for itself.
///
/// Measured on a 4070 (46 SMs) over the 88 distinct `Conv` geometries of
/// resnet50-qdq, yolov4 and yolov8n (`examples/conv_blocked`): every geometry
/// at or above 24 workgroups was faster blocked (1.20×–4.74×), every geometry
/// below it was slower (down to 0.50×), with a single 1.15× exception at 22.
/// The quantity that separates them is neither `P` nor `K` on its own but how
/// much of the machine the grid can fill — below roughly half the SM count the
/// bigger tile only costs.
///
/// The constant is therefore tied to this GPU's SM count. Vulkan core exposes
/// no portable way to query it (only vendor extensions such as
/// `VK_AMD_shader_core_properties` do), so it stays a measured constant until
/// `VkContext` can derive it from the device.
const WG_FLOOR: usize = 24;

/// Whether [`blocked`] is worth its 64×64 tile on this output geometry.
///
/// Blanket routing is worth 1.04× on resnet50 and 1.41× on yolov8n; gated on
/// this predicate the same kernel is worth 1.33× and 1.57×, and 2.58× on
/// yolov4 — in each case within 0.1% of picking the best kernel per shape by
/// hand. See `examples/conv_blocked`.
pub fn prefer_blocked(pixels: usize, c_out: usize) -> bool {
    workgroups(pixels, c_out) >= WG_FLOOR
}

/// Declarations shared by both variants.
const PRELUDE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
struct Push {
    total: u32, c_in: u32, c_out: u32, group: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    phb: u32, pwb: u32, dh: u32, dw: u32, gsi: u32, has_bias: u32,
    split: u32,
}
var<immediate> pc: Push;
"#;

/// Direct 1D/2D Conv (1D normalized to 2D with W=1): one thread per output element
/// output, with group/stride/pad/dilation.
pub fn direct() -> String {
    format!(
        "{PRELUDE}{}",
        r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.total) { return; }
    let ow = o % pc.w_out;
    let t1 = o / pc.w_out;
    let oh = t1 % pc.h_out;
    let t2 = t1 / pc.h_out;
    let m = t2 % pc.c_out;
    let bn = t2 / pc.c_out;
    let gso = pc.c_out / pc.group;
    let g = m / gso;
    var acc = 0.0;
    for (var cg = 0u; cg < pc.gsi; cg = cg + 1u) {
        let ic = g * pc.gsi + cg;
        for (var r = 0u; r < pc.kh; r = r + 1u) {
            let ih = i32(oh) * i32(pc.sh) - i32(pc.phb) + i32(r) * i32(pc.dh);
            if (ih < 0 || ih >= i32(pc.h_in)) { continue; }
            for (var s = 0u; s < pc.kw; s = s + 1u) {
                let iw = i32(ow) * i32(pc.sw) - i32(pc.pwb) + i32(s) * i32(pc.dw);
                if (iw < 0 || iw >= i32(pc.w_in)) { continue; }
                let xidx = ((bn * pc.c_in + ic) * pc.h_in + u32(ih)) * pc.w_in + u32(iw);
                let widx = ((m * pc.gsi + cg) * pc.kh + r) * pc.kw + s;
                acc = acc + x[xidx] * w[widx];
            }
        }
    }
    if (pc.has_bias != 0u) { acc = acc + bias[m]; }
    out[o] = acc;
}
"#
    )
}

/// Implicit GEMM for `group == 1`:
/// `out[C_out, P] = W[C_out, C_in·KH·KW] × X_im2col[C_in·KH·KW, P]`, with
/// `P = H_out·W_out`. The im2col matrix is never materialized: its
/// columns are indexed on the fly into `x`, so the read bandwidth remains
/// that of the original tensor.
pub fn implicit_gemm() -> String {
    format!(
        "{PRELUDE}{}",
        r#"
const TILE = 16u;
var<workgroup> w_tile: array<f32, 256>;
var<workgroup> x_tile: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ty = lid.y;
    let tx = lid.x;
    let m = wid.y * TILE + ty;           // output channel = row
    let p = wid.x * TILE + tx;           // output pixel = column
    let bn = wid.z;                      // batch image
    let pixels = pc.h_out * pc.w_out;
    let kdepth = pc.c_in * pc.kh * pc.kw;

    var acc = 0.0;
    let ntiles = (kdepth + TILE - 1u) / TILE;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        let kw_idx = t * TILE + tx;
        if (m < pc.c_out && kw_idx < kdepth) {
            w_tile[ty * TILE + tx] = w[m * kdepth + kw_idx];
        } else {
            w_tile[ty * TILE + tx] = 0.0;
        }
        let kx_idx = t * TILE + ty;
        var value = 0.0;
        if (p < pixels && kx_idx < kdepth) {
            let ksize = pc.kh * pc.kw;
            let ic = kx_idx / ksize;
            let rem = kx_idx % ksize;
            let r = rem / pc.kw;
            let s = rem % pc.kw;
            let oh = p / pc.w_out;
            let ow = p % pc.w_out;
            let ih = i32(oh) * i32(pc.sh) - i32(pc.phb) + i32(r) * i32(pc.dh);
            let iw = i32(ow) * i32(pc.sw) - i32(pc.pwb) + i32(s) * i32(pc.dw);
            // out of bounds = zero: this is the conv's implicit padding
            if (ih >= 0 && ih < i32(pc.h_in) && iw >= 0 && iw < i32(pc.w_in)) {
                value = x[((bn * pc.c_in + ic) * pc.h_in + u32(ih)) * pc.w_in + u32(iw)];
            }
        }
        x_tile[ty * TILE + tx] = value;
        workgroupBarrier();
        for (var i = 0u; i < TILE; i = i + 1u) {
            acc = acc + w_tile[ty * TILE + i] * x_tile[i * TILE + tx];
        }
        workgroupBarrier();
    }
    if (m >= pc.c_out || p >= pixels) { return; }
    if (pc.has_bias != 0u) { acc = acc + bias[m]; }
    out[(bn * pc.c_out + m) * pixels + p] = acc;
}
"#
    )
}

/// The same implicit GEMM on a 64×64 output tile with a 4×4 micro-tile held in
/// registers, for the geometries [`prefer_blocked`] accepts.
///
/// [`implicit_gemm`] keeps one output per thread, so its inner loop spends two
/// shared reads per FMA; here 8 reads feed 16 FMAs. That is the transformation
/// `MatMul` and `Gemm` already took, and on `Conv` it is worth anything from
/// 0.50× to 4.74× depending purely on how many workgroups the grid launches —
/// hence the predicate. `W` is read straight as the GEMM's `A`; the `B` side is
/// the im2col matrix, whose staging tile rebuilds each column from its `K`
/// index exactly as the 16×16 kernel does, so nothing is materialized here
/// either.
///
/// Accumulation order is identical to [`implicit_gemm`] — both walk `K` in
/// steps of 16 — so the two kernels agree bit for bit on all 88 geometries
/// measured, and routing between them cannot move a model's output.
pub fn blocked() -> String {
    blocked_body(false)
}

/// [`blocked`] with its `K` loop sliced across `wid.z`, writing one partial
/// image per slice for [`SPLIT_REDUCE`] to sum.
///
/// `wid.z` carries both the batch image and the slice — `bn = z / split`,
/// `slice = z % split` — since the grid has only three dimensions and the batch
/// already owned this one. The bias is deliberately not applied here: adding it
/// per slice would multiply it by `split`. It belongs to the reduction, the only
/// pass that sees a whole sum.
pub fn blocked_splitk() -> String {
    blocked_body(true)
}

fn blocked_body(split: bool) -> String {
    // the three lines the split-K variant changes: which slice of K this
    // workgroup walks, and where its partial goes
    let (batch, bounds, store) = if split {
        (
            "let bn = wid.z / pc.split;\n    let slice = wid.z % pc.split;",
            "let tper = (ntiles + pc.split - 1u) / pc.split;\n\
             \x20   let tstart = slice * tper;\n\
             \x20   var tend = tstart + tper;\n\
             \x20   if (tend > ntiles) { tend = ntiles; }",
            "out[slice * pc.total + (bn * pc.c_out + m) * pixels + p] = accv[j];",
        )
    } else {
        (
            "let bn = wid.z;",
            "let tstart = 0u;\n    let tend = ntiles;",
            "var v = accv[j];\n\
             \x20           if (pc.has_bias != 0u) { v = v + bias[m]; }\n\
             \x20           out[(bn * pc.c_out + m) * pixels + p] = v;",
        )
    };
    format!(
        "{PRELUDE}{}",
        r#"
const TILE = 64u;
const KSTEP = 16u;
var<workgroup> w_tile: array<f32, 1024>;
var<workgroup> x_tile: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.y * 16u + lid.x;
    let row0 = wid.y * TILE;          // first output channel of the block
    let col0 = wid.x * TILE;          // first output pixel of the block
    {batch}
    let pixels = pc.h_out * pc.w_out;
    let kdepth = pc.c_in * pc.kh * pc.kw;
    let ksize = pc.kh * pc.kw;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let ntiles = (kdepth + KSTEP - 1u) / KSTEP;
    {bounds}
    for (var t = tstart; t < tend; t = t + 1u) {
        let k0 = t * KSTEP;
        // --- stage W: 64 rows × 16 of K, 4 values per thread
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let gr = row0 + l / KSTEP;
            let gk = k0 + l % KSTEP;
            var v = 0.0;
            if (gr < pc.c_out && gk < kdepth) { v = w[gr * kdepth + gk]; }
            w_tile[l] = v;
        }
        // --- stage im2col: 16 of K × 64 pixels, rebuilt from the index
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let gk = k0 + l / TILE;
            let gc = col0 + l % TILE;
            var v = 0.0;
            if (gk < kdepth && gc < pixels) {
                let ic = gk / ksize;
                let rem = gk % ksize;
                let r = rem / pc.kw;
                let sx = rem % pc.kw;
                let oh = gc / pc.w_out;
                let ow = gc % pc.w_out;
                let ih = i32(oh) * i32(pc.sh) - i32(pc.phb) + i32(r) * i32(pc.dh);
                let iw = i32(ow) * i32(pc.sw) - i32(pc.pwb) + i32(sx) * i32(pc.dw);
                // out of bounds = zero: the conv's implicit padding
                if (ih >= 0 && ih < i32(pc.h_in) && iw >= 0 && iw < i32(pc.w_in)) {
                    v = x[((bn * pc.c_in + ic) * pc.h_in + u32(ih)) * pc.w_in + u32(iw)];
                }
            }
            x_tile[l] = v;
        }
        workgroupBarrier();
        // --- 4 scalars of W + 4 of im2col per 16 FMAs
        let arow = lid.y * 4u;
        let bcol = lid.x * 4u;
        for (var kk = 0u; kk < KSTEP; kk = kk + 1u) {
            let bo = kk * TILE + bcol;
            let bvec = vec4<f32>(x_tile[bo], x_tile[bo + 1u], x_tile[bo + 2u], x_tile[bo + 3u]);
            acc0 = fma(vec4<f32>(w_tile[(arow + 0u) * KSTEP + kk]), bvec, acc0);
            acc1 = fma(vec4<f32>(w_tile[(arow + 1u) * KSTEP + kk]), bvec, acc1);
            acc2 = fma(vec4<f32>(w_tile[(arow + 2u) * KSTEP + kk]), bvec, acc2);
            acc3 = fma(vec4<f32>(w_tile[(arow + 3u) * KSTEP + kk]), bvec, acc3);
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = row0 + lid.y * 4u + i;
        if (m >= pc.c_out) { continue; }
        var accv = acc0;
        if (i == 1u) { accv = acc1; }
        if (i == 2u) { accv = acc2; }
        if (i == 3u) { accv = acc3; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let p = col0 + lid.x * 4u + j;
            if (p >= pixels) { continue; }
            {store}
        }
    }
}
"#
        .replace("{batch}", batch)
        .replace("{bounds}", bounds)
        .replace("{store}", store)
    )
}

// ------------------------------------------------------------- split-K

pub const SPLIT_REDUCE_BINDINGS: u32 = 3;
/// Workgroups the split is sized to launch, on the 64×64 tile.
const SPLIT_TARGET_WGS: usize = 128;
/// Ceiling on the split, bounding the partials buffer at `32 ×` the output.
const SPLIT_MAX: usize = 32;
/// Smallest `K` worth splitting at all, and the smallest slice a split may
/// leave a workgroup — both one `KSTEP` block per staging round, four deep.
const SPLIT_MIN_K: usize = 256;
const SPLIT_MIN_K_PER_SLICE: usize = 64;

/// How many ways to split `K = C_in·KH·KW`, or `None` to dispatch one pass.
///
/// [`prefer_blocked`] answers a different question than this one, and both are
/// needed. `docs/resnet50-gap.md` measured that the 64×64 tile is worth ~1.9× on
/// resnet50's small-output convolutions **once the machine is full** and ~1.0×
/// at batch 1, and concluded no tile change reaches them. That was right about
/// the tile and wrong about the conclusion: the wide tile buys arithmetic
/// intensity by spending grid, and split-K is what buys the grid back. Neither
/// works alone — measured on those geometries, split-K on the 16×16 kernel is
/// **1.18×**, the 64×64 tile alone is ~1.0×, and together they are **3.8×**.
///
/// So a `Some` here means *both*: the 64×64 tile and this many slices, whatever
/// [`prefer_blocked`] would have said. `K` is the discriminator, not the
/// workgroup count — the geometries where splitting never pays
/// (`3→64 7×7`, `64→256 1×1`, `64→64 1×1`, `128→512 1×1`) are exactly those
/// with `K ≤ 147`, which have nothing to split.
///
/// Sized to land near `SPLIT_TARGET_WGS` workgroups, as
/// `matmul_fp32::gemv_split` does, and calibrated the same way: on this GPU's
/// 46 SMs, with `examples/conv_splitk`. Measured 1.3× to 4.7× per geometry and
/// **2.45× over all 53 `Conv` nodes** of resnet50-qdq.
pub fn split_k(pixels: usize, c_out: usize, kdepth: usize) -> Option<u32> {
    split_k_for_target(pixels, c_out, kdepth, SPLIT_TARGET_WGS)
}

fn split_k_for_target(
    pixels: usize,
    c_out: usize,
    kdepth: usize,
    target_workgroups: usize,
) -> Option<u32> {
    if kdepth < SPLIT_MIN_K {
        return None;
    }
    let base = workgroups(pixels, c_out);
    let split = target_workgroups
        .div_ceil(base.max(1))
        .min(SPLIT_MAX)
        .min(kdepth / SPLIT_MIN_K_PER_SLICE);
    (split > 1).then_some(split as u32)
}

/// Reproduces the currently shipped floating-point `Conv` routing policy.
///
/// This is the control tactic against which future generated candidates are
/// compared. Grouped and depthwise convolutions stay direct; a profitable
/// split takes precedence over the blocked-tile predicate, exactly as in the
/// original interpreter ladder.
pub fn control_tactic(group: usize, pixels: usize, c_out: usize, kdepth: usize) -> Tactic {
    if group != 1 {
        return Tactic::Direct {
            workgroup_size: DIRECT_WORKGROUP_SIZE,
        };
    }
    if let Some(split) = split_k(pixels, c_out, kdepth) {
        return Tactic::SplitK {
            output_tile: BLOCKED_TILE_SIZE,
            k_step: BLOCKED_K_STEP,
            micro_tile: BLOCKED_MICRO_TILE_SIZE,
            split,
        };
    }
    if prefer_blocked(pixels, c_out) {
        Tactic::Blocked {
            output_tile: BLOCKED_TILE_SIZE,
            k_step: BLOCKED_K_STEP,
            micro_tile: BLOCKED_MICRO_TILE_SIZE,
        }
    } else {
        Tactic::ImplicitGemm {
            output_tile: TILE_SIZE,
            k_step: TILE_SIZE,
        }
    }
}

/// Compact route proposed by the first census-wide search.
///
/// A 32 tile avoids channel padding for very narrow outputs, while a 128 tile
/// is used only when K is deep enough and its own grid still reaches the
/// measured occupancy floor. Split-K overrides are deliberately exact: the
/// earlier broad `K >= 2304` rule won in isolated dispatches but regressed the
/// Tier-2 resnet50-qdq profile. These signatures are the census points whose
/// interleaved focused-search winner had a material margin; every other shape
/// retains the shipped control.
pub fn tuned_tactic(group: usize, pixels: usize, c_out: usize, kdepth: usize) -> Tactic {
    if group != 1 {
        return control_tactic(group, pixels, c_out, kdepth);
    }
    let control = control_tactic(group, pixels, c_out, kdepth);
    let tuned_split = match (pixels, c_out, kdepth) {
        (169, 512, 4608) => Some(32),
        (169, 1024, 4608) => Some(16),
        (676, 512, 2304) => Some(8),
        (169, 512, 2304) => Some(16),
        (169, 1024, 1024) => Some(8),
        (10_816, 64, 576) => Some(4),
        _ => None,
    };
    if let (Tactic::SplitK { .. }, Some(split)) = (control, tuned_split) {
        return Tactic::SplitK {
            output_tile: BLOCKED_TILE_SIZE,
            k_step: BLOCKED_K_STEP,
            micro_tile: BLOCKED_MICRO_TILE_SIZE,
            split,
        };
    }
    if matches!(control, Tactic::Blocked { .. }) {
        if c_out <= 32 {
            return Tactic::Blocked {
                output_tile: 32,
                k_step: BLOCKED_K_STEP,
                micro_tile: BLOCKED_MICRO_TILE_SIZE,
            };
        }
        if c_out >= 128 && kdepth >= 256 && pixels.div_ceil(128) * c_out.div_ceil(128) >= WG_FLOOR {
            return Tactic::Blocked {
                output_tile: 128,
                k_step: BLOCKED_K_STEP,
                micro_tile: BLOCKED_MICRO_TILE_SIZE,
            };
        }
    }
    control
}

/// Sums the `split` partial images and applies the bias.
pub const SPLIT_REDUCE: &str = r#"
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
struct Push {
    total: u32, c_in: u32, c_out: u32, group: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    phb: u32, pwb: u32, dh: u32, dw: u32, gsi: u32, has_bias: u32,
    split: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.total) { return; }
    var acc = 0.0;
    for (var s = 0u; s < pc.split; s = s + 1u) {
        acc = acc + partials[s * pc.total + o];
    }
    if (pc.has_bias != 0u) {
        // o indexes [N, C_out, P]: the channel is what selects the bias
        acc = acc + bias[(o / (pc.h_out * pc.w_out)) % pc.c_out];
    }
    out[o] = acc;
}
"#;

#[cfg(test)]
mod tests {
    use super::{
        BLOCKED_K_STEP, BLOCKED_MICRO_TILE_SIZE, BLOCKED_TILE_SIZE, DIRECT_WORKGROUP_SIZE,
        TILE_SIZE, Tactic, TacticGeometry, TacticLimits, TacticRejection, candidate_space,
        control_tactic, focused_search_space, generated_source, prefer_blocked,
        reduced_exhaustive_space, tuned_tactic,
    };

    fn representative_geometry() -> TacticGeometry {
        TacticGeometry {
            batch: 1,
            group: 1,
            pixels: 49,
            c_out: 512,
            kdepth: 512,
            total: 49 * 512,
        }
    }

    fn representative_limits() -> TacticLimits {
        TacticLimits {
            max_workgroup_count: [65_535; 3],
            max_workgroup_invocations: 1024,
            max_workgroup_size: [1024, 1024, 64],
            max_shared_memory_bytes: 32 * 1024,
            max_storage_buffer_bytes: 1 << 30,
            max_scratch_bytes: 1 << 30,
        }
    }

    #[test]
    fn sources_compile() {
        for source in [super::direct(), super::implicit_gemm(), super::blocked()] {
            vk_compute::compile_wgsl(&source).expect("shader Conv f32 valido");
        }
    }

    /// The measured split from `examples/conv_blocked`: the geometries below
    /// the floor were 0.50×–0.83× on a 4070, the ones at or above it
    /// 1.20×–4.74×. These are real nodes of resnet50-qdq, yolov4 and yolov8n.
    #[test]
    fn the_predicate_matches_what_was_measured() {
        // rejected: too few workgroups to fill the machine
        assert!(!prefer_blocked(49, 512)); //  8 wg — resnet50 512→512 3x3 @7²
        assert!(!prefer_blocked(196, 256)); // 16 wg — resnet50 256→256 3x3 @14²
        assert!(!prefer_blocked(400, 64)); //  7 wg — yolov8n 64→64 3x3 @20²
        assert!(!prefer_blocked(169, 255)); // 12 wg — yolov4 1024→255 1x1 @13²
        assert!(!prefer_blocked(16, 1)); //  1 wg — yolov8n 16→1 1x1 @4²

        // accepted, starting exactly at the floor
        assert!(prefer_blocked(169, 512)); //  24 wg — yolov4 512→512 3x3 @13²
        assert!(prefer_blocked(784, 128)); //  26 wg — resnet50 128→128 3x3 @28²
        assert!(prefer_blocked(49, 2048)); //  32 wg — resnet50 512→2048 1x1 @7²
        assert!(prefer_blocked(43264, 64)); // 676 wg — yolov4 64→64 1x1 @208²
        assert!(prefer_blocked(173056, 32)); // 2704 wg — yolov4 3→32 3x3 @416²
    }

    /// Small `P` alone does not reject: enough output channels make the grid
    /// wide even on a 7×7 feature map, and those geometries did gain (1.57×).
    #[test]
    fn channels_can_carry_a_tiny_feature_map() {
        assert!(!prefer_blocked(49, 512));
        assert!(prefer_blocked(49, 2048));
    }

    #[test]
    fn control_tactic_preserves_the_shipped_routing_ladder() {
        assert_eq!(
            control_tactic(32, 784, 32, 9),
            Tactic::Direct {
                workgroup_size: DIRECT_WORKGROUP_SIZE,
            }
        );
        assert_eq!(
            control_tactic(1, 49, 512, 128),
            Tactic::ImplicitGemm {
                output_tile: TILE_SIZE,
                k_step: TILE_SIZE,
            }
        );
        assert_eq!(
            control_tactic(1, 49, 2048, 128),
            Tactic::Blocked {
                output_tile: BLOCKED_TILE_SIZE,
                k_step: BLOCKED_K_STEP,
                micro_tile: BLOCKED_MICRO_TILE_SIZE,
            }
        );
        let Tactic::SplitK {
            output_tile,
            k_step,
            micro_tile,
            split,
        } = control_tactic(1, 49, 512, 512)
        else {
            panic!("large reduction must select the shipped split-K tactic");
        };
        assert_eq!(output_tile, BLOCKED_TILE_SIZE);
        assert_eq!(k_step, BLOCKED_K_STEP);
        assert_eq!(micro_tile, BLOCKED_MICRO_TILE_SIZE);
        assert!(split > 1);
    }

    #[test]
    fn tactic_parameters_drive_the_existing_dispatch_shapes() {
        let direct = control_tactic(8, 49, 64, 9);
        assert_eq!(direct.dispatch_grid(3136, 49, 64, 1), [13, 1, 1]);

        let split = control_tactic(1, 49, 512, 512);
        assert_eq!(
            split.dispatch_grid(49 * 512, 49, 512, 2),
            [1, 8, 2 * split.split()]
        );
    }

    #[test]
    fn persistent_tactic_encoding_round_trips_and_rejects_unknown_fields() {
        for tactic in [
            Tactic::Direct {
                workgroup_size: 256,
            },
            Tactic::ImplicitGemm {
                output_tile: 16,
                k_step: 16,
            },
            Tactic::Blocked {
                output_tile: 32,
                k_step: 16,
                micro_tile: 4,
            },
            Tactic::SplitK {
                output_tile: 64,
                k_step: 16,
                micro_tile: 4,
                split: 8,
            },
        ] {
            let (id, parameters) = tactic.persistent_parts();
            assert_eq!(Tactic::from_persistent(&id, &parameters), Some(tactic));
            let mut unknown = parameters.clone();
            unknown.insert("external_shader".into(), 1);
            assert_eq!(Tactic::from_persistent(&id, &unknown), None);
        }
    }

    #[test]
    fn shipped_tactics_are_viable_and_accept_edge_padding() {
        let limits = representative_limits();
        let geometry = representative_geometry();
        assert!(control_tactic(1, 49, 512, 512).is_viable(geometry, limits));

        let padded = TacticGeometry {
            pixels: 50,
            c_out: 513,
            total: 50 * 513,
            ..geometry
        };
        let blocked = Tactic::Blocked {
            output_tile: BLOCKED_TILE_SIZE,
            k_step: BLOCKED_K_STEP,
            micro_tile: BLOCKED_MICRO_TILE_SIZE,
        };
        assert!(blocked.is_viable(padded, limits));

        let grouped = TacticGeometry {
            group: 64,
            c_out: 64,
            kdepth: 9,
            total: 49 * 64,
            ..geometry
        };
        assert!(control_tactic(64, 49, 64, 9).is_viable(grouped, limits));
    }

    #[test]
    fn geometry_and_parameter_rejections_are_typed() {
        let limits = representative_limits();
        let geometry = representative_geometry();
        let blocked = Tactic::Blocked {
            output_tile: BLOCKED_TILE_SIZE,
            k_step: BLOCKED_K_STEP,
            micro_tile: BLOCKED_MICRO_TILE_SIZE,
        };

        assert_eq!(
            blocked.viability(
                TacticGeometry {
                    group: 2,
                    ..geometry
                },
                limits
            ),
            Err(TacticRejection::GroupedGemm)
        );
        assert_eq!(
            Tactic::Direct {
                workgroup_size: DIRECT_WORKGROUP_SIZE,
            }
            .viability(
                TacticGeometry {
                    group: 3,
                    ..geometry
                },
                limits,
            ),
            Err(TacticRejection::InvalidGroup)
        );
        assert_eq!(
            Tactic::Blocked {
                output_tile: 63,
                k_step: 16,
                micro_tile: 4,
            }
            .viability(geometry, limits),
            Err(TacticRejection::InvalidParameters)
        );
        assert_eq!(
            Tactic::SplitK {
                output_tile: 64,
                k_step: 16,
                micro_tile: 4,
                split: 33,
            }
            .viability(geometry, limits),
            Err(TacticRejection::EmptySplit)
        );
        assert_eq!(
            blocked.viability(
                TacticGeometry {
                    pixels: 0,
                    ..geometry
                },
                limits
            ),
            Err(TacticRejection::EmptyGeometry)
        );
        assert_eq!(
            blocked.viability(
                TacticGeometry {
                    total: 1,
                    ..geometry
                },
                limits
            ),
            Err(TacticRejection::InconsistentOutputElements)
        );
        assert_eq!(
            blocked.viability(
                TacticGeometry {
                    batch: u32::MAX,
                    pixels: u32::MAX,
                    c_out: u32::MAX,
                    total: u64::MAX,
                    ..geometry
                },
                limits,
            ),
            Err(TacticRejection::ArithmeticOverflow)
        );
    }

    #[test]
    fn device_and_scratch_limits_reject_before_execution() {
        let geometry = representative_geometry();
        let tactic = control_tactic(1, 49, 512, 512);
        let limits = representative_limits();

        assert_eq!(
            tactic.viability(
                geometry,
                TacticLimits {
                    max_workgroup_invocations: 255,
                    ..limits
                },
            ),
            Err(TacticRejection::WorkgroupLimit)
        );
        assert_eq!(
            tactic.viability(
                geometry,
                TacticLimits {
                    max_shared_memory_bytes: 8191,
                    ..limits
                },
            ),
            Err(TacticRejection::SharedMemoryLimit)
        );
        assert_eq!(
            tactic.viability(
                geometry,
                TacticLimits {
                    max_workgroup_count: [65_535, 7, 65_535],
                    ..limits
                },
            ),
            Err(TacticRejection::GridLimit)
        );

        let scratch_bytes = geometry.total * u64::from(tactic.split()) * 4;
        assert_eq!(
            tactic.viability(
                geometry,
                TacticLimits {
                    max_storage_buffer_bytes: scratch_bytes - 1,
                    ..limits
                },
            ),
            Err(TacticRejection::StorageBufferLimit)
        );
        assert_eq!(
            tactic.viability(
                geometry,
                TacticLimits {
                    max_scratch_bytes: scratch_bytes - 1,
                    ..limits
                },
            ),
            Err(TacticRejection::ScratchBudget)
        );
    }

    #[test]
    fn vulkan_limits_conversion_keeps_policy_separate() {
        let vulkan = vk_compute::ComputeLimits {
            max_workgroup_count: [11, 12, 13],
            max_workgroup_invocations: 14,
            max_workgroup_size: [15, 16, 17],
            max_shared_memory_bytes: 18,
            max_storage_buffer_bytes: 19,
        };
        let limits = TacticLimits::from_vulkan(&vulkan, 20);
        assert_eq!(limits.max_workgroup_count, [11, 12, 13]);
        assert_eq!(limits.max_workgroup_invocations, 14);
        assert_eq!(limits.max_workgroup_size, [15, 16, 17]);
        assert_eq!(limits.max_shared_memory_bytes, 18);
        assert_eq!(limits.max_storage_buffer_bytes, 19);
        assert_eq!(limits.max_scratch_bytes, 20);
    }

    #[test]
    fn candidate_space_is_complete_unique_and_contains_control_parameters() {
        use std::collections::HashSet;

        let candidates = candidate_space(1);
        assert_eq!(candidates.len(), 2580);
        assert_eq!(
            candidates.iter().copied().collect::<HashSet<_>>().len(),
            2580
        );

        for geometry in [
            representative_geometry(),
            TacticGeometry {
                pixels: 196,
                c_out: 256,
                kdepth: 2304,
                total: 196 * 256,
                ..representative_geometry()
            },
        ] {
            let control = control_tactic(
                geometry.group as usize,
                geometry.pixels as usize,
                geometry.c_out as usize,
                geometry.kdepth as usize,
            );
            assert!(candidates.contains(&control));
        }

        let grouped = candidate_space(64);
        assert_eq!(grouped.len(), 5);
        assert!(
            grouped
                .iter()
                .all(|tactic| matches!(tactic, Tactic::Direct { .. }))
        );
    }

    #[test]
    fn generated_shipped_points_are_byte_exact() {
        for (tactic, expected) in [
            (
                Tactic::Direct {
                    workgroup_size: DIRECT_WORKGROUP_SIZE,
                },
                super::direct(),
            ),
            (
                Tactic::ImplicitGemm {
                    output_tile: TILE_SIZE,
                    k_step: TILE_SIZE,
                },
                super::implicit_gemm(),
            ),
            (
                Tactic::Blocked {
                    output_tile: BLOCKED_TILE_SIZE,
                    k_step: BLOCKED_K_STEP,
                    micro_tile: BLOCKED_MICRO_TILE_SIZE,
                },
                super::blocked(),
            ),
            (
                Tactic::SplitK {
                    output_tile: BLOCKED_TILE_SIZE,
                    k_step: BLOCKED_K_STEP,
                    micro_tile: BLOCKED_MICRO_TILE_SIZE,
                    split: 8,
                },
                super::blocked_splitk(),
            ),
        ] {
            assert_eq!(generated_source(tactic).as_deref(), Some(expected.as_str()));
        }
    }

    #[test]
    fn reduced_space_is_unique_and_every_source_compiles() {
        use std::collections::HashSet;

        let candidates = reduced_exhaustive_space();
        assert_eq!(candidates.len(), 125);
        assert_eq!(
            candidates.iter().copied().collect::<HashSet<_>>().len(),
            candidates.len()
        );
        for tactic in candidates {
            let Some(source) = generated_source(tactic) else {
                panic!("reduced tactic has no generator: {tactic:?}");
            };
            vk_compute::compile_wgsl(&source).expect("reduced Conv source must compile");
        }
    }

    #[test]
    fn focused_space_has_the_documented_budget() {
        use std::collections::HashSet;

        let candidates = focused_search_space();
        assert_eq!(candidates.len(), 21);
        assert_eq!(
            candidates.iter().copied().collect::<HashSet<_>>().len(),
            candidates.len()
        );
    }

    #[test]
    fn tuned_route_keeps_fallbacks_and_exposes_the_promoted_tile_rules() {
        assert_eq!(
            tuned_tactic(64, 6400, 64, 9),
            control_tactic(64, 6400, 64, 9)
        );
        assert_eq!(
            tuned_tactic(1, 173_056, 32, 27),
            Tactic::Blocked {
                output_tile: 32,
                k_step: BLOCKED_K_STEP,
                micro_tile: BLOCKED_MICRO_TILE_SIZE,
            }
        );
        assert_eq!(
            tuned_tactic(1, 2704, 256, 1152),
            Tactic::Blocked {
                output_tile: 128,
                k_step: BLOCKED_K_STEP,
                micro_tile: BLOCKED_MICRO_TILE_SIZE,
            }
        );
        assert_eq!(
            tuned_tactic(1, 196, 256, 2304),
            control_tactic(1, 196, 256, 2304)
        );
        assert_eq!(
            tuned_tactic(1, 169, 512, 4608),
            Tactic::SplitK {
                output_tile: BLOCKED_TILE_SIZE,
                k_step: BLOCKED_K_STEP,
                micro_tile: BLOCKED_MICRO_TILE_SIZE,
                split: 32,
            }
        );
        assert_eq!(
            tuned_tactic(1, 196, 256, 2304),
            control_tactic(1, 196, 256, 2304)
        );
    }
}
