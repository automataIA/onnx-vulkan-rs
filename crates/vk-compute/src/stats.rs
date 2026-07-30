//! Profiler (enabled with `VULKAN_EP_STATS=1`).
//!
//! Measures **real GPU time per op type** via Vulkan timestamp queries
//! (timestamp after each dispatch; barriers between dispatches make them
//! attributable). Also benchmarks flush wall-clock time (= GPU sync) and
//! transferred bytes. Percentage breakdown (Pareto) printed at `OnRunEnd`.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VULKAN_EP_STATS").is_some())
}

thread_local! {
    /// Current op set by kernel at `Compute` start; read by stream
    /// to attribute dispatch timestamps.
    static CURRENT_OP: Cell<&'static str> = const { Cell::new("?") };
    /// First pipeline key of the node being executed, i.e. the kernel that does
    /// the node's work. It is not `CURRENT_OP`: a split-K `Conv` sets
    /// `Conv_split` and then `Conv_split_reduce`, and the node's FLOPs belong to
    /// the first. The reduction keeps its own milliseconds and no work — which
    /// is the correct reading, since its cost is the price paid for the split.
    static PRIMARY_OP: Cell<Option<&'static str>> = const { Cell::new(None) };
}

pub fn set_op(name: &'static str) {
    if enabled() {
        CURRENT_OP.with(|c| c.set(name));
        PRIMARY_OP.with(|c| {
            if c.get().is_none() {
                c.set(Some(name));
            }
        });
    }
}

/// Starts the attribution window of one node: `set_op` after this call names
/// the kernel the node's work is charged to.
pub fn begin_node() {
    if enabled() {
        CURRENT_OP.with(|c| c.set("compile"));
        PRIMARY_OP.with(|c| c.set(None));
    }
}

/// Kernel the current node's work belongs to; `"compile"` when no pipeline was
/// used (host-side ops, uploads).
pub fn primary_op() -> &'static str {
    PRIMARY_OP.with(|c| c.get()).unwrap_or("compile")
}

pub fn current_op() -> &'static str {
    CURRENT_OP.with(|c| c.get())
}

/// GPU ns and dispatch count per op type.
static GPU_TIME: Mutex<Option<HashMap<&'static str, (u64, u64)>>> = Mutex::new(None);
/// Analytic FLOPs and compulsory bytes per op type, summed over the nodes
/// charged to it. Computed by the caller from the graph — the profiler can
/// time a dispatch but has no way to know how much work it represents.
static WORK: Mutex<Option<HashMap<&'static str, (u64, u64)>>> = Mutex::new(None);
/// Cumulative wall-clock of flushes (GPU sync), ns.
static FLUSH_WALL_NS: AtomicU64 = AtomicU64::new(0);
static FLUSHES: AtomicU64 = AtomicU64::new(0);
static UP_BYTES: AtomicU64 = AtomicU64::new(0);
static DOWN_BYTES: AtomicU64 = AtomicU64::new(0);
/// Queue submissions. **Not the same as `FLUSHES`**: the stream's `flush` is
/// one submit, but the one-shot `run_commands` path is another, and `FLUSHES`
/// is only recorded while profiling is on. A submit is counted wherever
/// `vkQueueSubmit` is called, so `submits > flushes` is the expected reading
/// and the difference is the one-shot path.
static SUBMITS: AtomicU64 = AtomicU64::new(0);
/// Transfer *counts*, next to the byte totals: 1×100 MB and 1000×100 KB are
/// the same `UP_BYTES` and very different costs.
static UPLOADS: AtomicU64 = AtomicU64::new(0);
static DOWNLOADS: AtomicU64 = AtomicU64::new(0);
/// Device-local storage buffers actually allocated, and requests served from
/// the pool instead. The ratio is what says whether `StoragePool` works.
static ALLOCS: AtomicU64 = AtomicU64::new(0);
/// Dispatches recorded into the stream, for probes that attribute them.
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
static POOL_HITS: AtomicU64 = AtomicU64::new(0);
/// Live device-local bytes and their peak: the memory the graph actually
/// holds, not what it allocated. The gap between the two indicates how
/// effective buffer reuse (`StoragePool`) is.
static STORAGE_LIVE: AtomicU64 = AtomicU64::new(0);
static STORAGE_PEAK: AtomicU64 = AtomicU64::new(0);

pub fn record_storage_alloc(bytes: u64) {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    let live = STORAGE_LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    STORAGE_PEAK.fetch_max(live, Ordering::Relaxed);
}

/// A storage-buffer request served by the pool: no allocation, no change to
/// live bytes (the buffer never left the accounting when it was pooled).
pub fn record_storage_pool_hit() {
    POOL_HITS.fetch_add(1, Ordering::Relaxed);
}

/// One `vkQueueSubmit`, from whichever path issued it.
pub fn record_submit() {
    SUBMITS.fetch_add(1, Ordering::Relaxed);
}

pub fn record_storage_free(bytes: u64) {
    STORAGE_LIVE.fetch_sub(bytes, Ordering::Relaxed);
}

/// Peak of allocated device-local bytes, without resetting it.
pub fn storage_peak_bytes() -> u64 {
    STORAGE_PEAK.load(Ordering::Relaxed)
}

pub fn reset_storage_peak() {
    STORAGE_PEAK.store(STORAGE_LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
}

pub fn record_gpu(op: &'static str, ns: u64) {
    let mut g = GPU_TIME.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    let e = map.entry(op).or_insert((0, 0));
    e.0 += ns;
    e.1 += 1;
}

/// Work of one node, charged to the kernel that ran it (`primary_op`).
pub fn record_work(op: &'static str, flops: u64, bytes: u64) {
    let mut w = WORK.lock().unwrap();
    let map = w.get_or_insert_with(HashMap::new);
    let e = map.entry(op).or_insert((0, 0));
    e.0 += flops;
    e.1 += bytes;
}

/// Host time spent recording commands, as opposed to waiting for them.
///
/// The decode step of an LLM is one flush, so its wall clock is GPU compute plus
/// whatever the host spent getting there, and that term is invisible in the
/// numbers above: `sync/overhead` charges it to the fence. This separates the
/// part a pre-recorded command buffer could remove (descriptor writes and
/// `vkCmd*`) from the part only a shorter interpreter path could — the rest of
/// the step's host time is graph walking, shape math and allocation.
static RECORD_NS: AtomicU64 = AtomicU64::new(0);
static RECORDED: AtomicU64 = AtomicU64::new(0);

pub fn record_recording(ns: u64) {
    RECORD_NS.fetch_add(ns, Ordering::Relaxed);
    RECORDED.fetch_add(1, Ordering::Relaxed);
}

pub fn record_flush(wall_ns: u64) {
    FLUSH_WALL_NS.fetch_add(wall_ns, Ordering::Relaxed);
    FLUSHES.fetch_add(1, Ordering::Relaxed);
}

pub fn record_up(bytes: u64) {
    UP_BYTES.fetch_add(bytes, Ordering::Relaxed);
    UPLOADS.fetch_add(1, Ordering::Relaxed);
}

pub fn record_down(bytes: u64) {
    DOWN_BYTES.fetch_add(bytes, Ordering::Relaxed);
    DOWNLOADS.fetch_add(1, Ordering::Relaxed);
}

/// Prints Pareto and resets counters (called at `OnRunEnd`).
pub fn dump_and_reset() {
    if !enabled() {
        return;
    }
    let mut g = GPU_TIME.lock().unwrap();
    let map = g.take().unwrap_or_default();
    let work = WORK.lock().unwrap().take().unwrap_or_default();
    let flush_wall = FLUSH_WALL_NS.swap(0, Ordering::Relaxed);
    let flushes = FLUSHES.swap(0, Ordering::Relaxed);
    let up = UP_BYTES.swap(0, Ordering::Relaxed);
    let down = DOWN_BYTES.swap(0, Ordering::Relaxed);
    // swapped before the early return below: they are per-run counters, and a
    // run with no GPU work still has to leave them at zero for the next one
    let submits = SUBMITS.swap(0, Ordering::Relaxed);
    let uploads = UPLOADS.swap(0, Ordering::Relaxed);
    let downloads = DOWNLOADS.swap(0, Ordering::Relaxed);
    let allocs = ALLOCS.swap(0, Ordering::Relaxed);
    let pool_hits = POOL_HITS.swap(0, Ordering::Relaxed);
    let record_ns = RECORD_NS.swap(0, Ordering::Relaxed);
    let recorded = RECORDED.swap(0, Ordering::Relaxed);

    let gpu_total: u64 = map.values().map(|(ns, _)| *ns).sum();
    if gpu_total == 0 && flush_wall == 0 {
        return;
    }

    let mut rows: Vec<(&str, u64, u64)> = map.iter().map(|(k, (ns, c))| (*k, *ns, *c)).collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));

    log::info!("── VulkanEP profile (Pareto by GPU time) ──");
    for (op, ns, count) in &rows {
        // `gflop`/`gb` are the Roofline numerators: the analytic work of the
        // nodes charged to this kernel. Absent when the work model does not
        // cover the op — a missing column is honest, a zero would not be.
        let w = match work.get(op) {
            Some((flops, bytes)) => format!(
                "  gflop={:.4} gb={:.4}",
                *flops as f64 / 1e9,
                *bytes as f64 / 1e9
            ),
            None => String::new(),
        };
        log::info!(
            "  {:<22} {:>8.3} ms  {:>5.1}%  ({} dispatch){}",
            op,
            *ns as f64 / 1e6,
            *ns as f64 / gpu_total as f64 * 100.0,
            count,
            w,
        );
    }
    let sync_ns = flush_wall.saturating_sub(gpu_total);
    log::info!(
        "  {:<22} {:>8.3} ms         (flush wall)",
        "TOTAL GPU compute",
        gpu_total as f64 / 1e6
    );
    if recorded > 0 {
        log::info!(
            "  {:<22} {:>8.3} ms         ({recorded} commands, host side)",
            "recording",
            record_ns as f64 / 1e6
        );
    }
    log::info!(
        "  sync/overhead ~{:.3} ms across {} flushes; transfer up {:.1} MB / down {:.1} MB",
        sync_ns as f64 / 1e6,
        flushes,
        up as f64 / 1e6,
        down as f64 / 1e6,
    );
    // both numbers, because they answer different questions: the peak is what
    // the device had to hold, `live` is what is still held now the run is over.
    // A generation loop's `live` must come back to the same value every step —
    // a peak that grows can just be a longer sequence, a `live` that grows is a
    // buffer nobody freed.
    let live = STORAGE_LIVE.load(Ordering::Relaxed);
    log::info!(
        "  tensor VRAM peak {:.1} MB, live {:.1} MB",
        STORAGE_PEAK.swap(live, Ordering::Relaxed) as f64 / 1e6,
        live as f64 / 1e6,
    );
    // Counted, not derived from the timings above: an experiment loop needs a
    // metric that does not move between two identical runs, and these do not.
    log::info!(
        "  counters: submit={} upload={} download={} alloc={} pool-hit={}",
        submits,
        uploads,
        downloads,
        allocs,
        pool_hits,
    );
}

/// Storage-buffer allocations so far (probe hook, see interp's ALLOC_PROBE).
pub fn allocs() -> u64 {
    ALLOCS.load(Ordering::Relaxed)
}

/// Dispatches recorded into the stream so far (probe hook, see the interpreter's
/// `NODE_PROBE`). Counted unconditionally: one relaxed increment against a
/// command-buffer recording is not measurable.
pub fn dispatches() -> u64 {
    DISPATCHES.load(Ordering::Relaxed)
}

pub(crate) fn record_dispatch_count() {
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
}
