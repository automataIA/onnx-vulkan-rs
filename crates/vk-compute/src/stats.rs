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
}

pub fn set_op(name: &'static str) {
    if enabled() {
        CURRENT_OP.with(|c| c.set(name));
    }
}

pub fn current_op() -> &'static str {
    CURRENT_OP.with(|c| c.get())
}

/// GPU ns and dispatch count per op type.
static GPU_TIME: Mutex<Option<HashMap<&'static str, (u64, u64)>>> = Mutex::new(None);
/// Cumulative wall-clock of flushes (GPU sync), ns.
static FLUSH_WALL_NS: AtomicU64 = AtomicU64::new(0);
static FLUSHES: AtomicU64 = AtomicU64::new(0);
static UP_BYTES: AtomicU64 = AtomicU64::new(0);
static DOWN_BYTES: AtomicU64 = AtomicU64::new(0);
/// Live device-local bytes and their peak: the memory the graph actually
/// holds, not what it allocated. The gap between the two indicates how
/// effective buffer reuse (`StoragePool`) is.
static STORAGE_LIVE: AtomicU64 = AtomicU64::new(0);
static STORAGE_PEAK: AtomicU64 = AtomicU64::new(0);

pub fn record_storage_alloc(bytes: u64) {
    let live = STORAGE_LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    STORAGE_PEAK.fetch_max(live, Ordering::Relaxed);
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

pub fn record_flush(wall_ns: u64) {
    FLUSH_WALL_NS.fetch_add(wall_ns, Ordering::Relaxed);
    FLUSHES.fetch_add(1, Ordering::Relaxed);
}

pub fn record_up(bytes: u64) {
    UP_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

pub fn record_down(bytes: u64) {
    DOWN_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Prints Pareto and resets counters (called at `OnRunEnd`).
pub fn dump_and_reset() {
    if !enabled() {
        return;
    }
    let mut g = GPU_TIME.lock().unwrap();
    let map = g.take().unwrap_or_default();
    let flush_wall = FLUSH_WALL_NS.swap(0, Ordering::Relaxed);
    let flushes = FLUSHES.swap(0, Ordering::Relaxed);
    let up = UP_BYTES.swap(0, Ordering::Relaxed);
    let down = DOWN_BYTES.swap(0, Ordering::Relaxed);

    let gpu_total: u64 = map.values().map(|(ns, _)| *ns).sum();
    if gpu_total == 0 && flush_wall == 0 {
        return;
    }

    let mut rows: Vec<(&str, u64, u64)> = map.iter().map(|(k, (ns, c))| (*k, *ns, *c)).collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));

    log::info!("── VulkanEP profile (Pareto by GPU time) ──");
    for (op, ns, count) in &rows {
        log::info!(
            "  {:<22} {:>8.3} ms  {:>5.1}%  ({} dispatch)",
            op,
            *ns as f64 / 1e6,
            *ns as f64 / gpu_total as f64 * 100.0,
            count
        );
    }
    let sync_ns = flush_wall.saturating_sub(gpu_total);
    log::info!(
        "  {:<22} {:>8.3} ms         (flush wall)",
        "TOTAL GPU compute",
        gpu_total as f64 / 1e6
    );
    log::info!(
        "  sync/overhead ~{:.3} ms across {} flushes; transfer up {:.1} MB / down {:.1} MB",
        sync_ns as f64 / 1e6,
        flushes,
        up as f64 / 1e6,
        down as f64 / 1e6,
    );
    log::info!(
        "  picco VRAM tensori {:.1} MB",
        STORAGE_PEAK.swap(STORAGE_LIVE.load(Ordering::Relaxed), Ordering::Relaxed) as f64 / 1e6,
    );
}
