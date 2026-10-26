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

/// Host time per named phase of a run, and how many times it ran: uploads,
/// readbacks, the retirement after a fence, the interpreter loop. What is left
/// of a wall clock once GPU compute is taken out has to be found somewhere,
/// and these say where. They nest: `execute nodes` contains the uploads,
/// recording and any flush its nodes issue, and a caller's span (the
/// runner's `facade run`) contains everything its call did.
static SPANS: Mutex<Option<HashMap<&'static str, (u64, u64)>>> = Mutex::new(None);

pub fn record_span(name: &'static str, ns: u64) {
    let mut s = SPANS.lock().unwrap();
    let e = s
        .get_or_insert_with(HashMap::new)
        .entry(name)
        .or_insert((0, 0));
    e.0 += ns;
    e.1 += 1;
}

/// A phase timed until dropped; free when profiling is off.
pub struct Span(Option<(&'static str, std::time::Instant)>);

pub fn span(name: &'static str) -> Span {
    Span(enabled().then(|| (name, std::time::Instant::now())))
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some((name, start)) = self.0 {
            record_span(name, start.elapsed().as_nanos() as u64);
        }
    }
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
    let mut spans: Vec<_> = SPANS
        .lock()
        .unwrap()
        .take()
        .unwrap_or_default()
        .into_iter()
        .collect();
    spans.sort_unstable();

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
    for (name, (ns, count)) in &spans {
        log::info!(
            "  {:<22} {:>8.3} ms         ({count}x, host span)",
            name,
            *ns as f64 / 1e6
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

/// GPU timestamps of one inference as the `VULKAN_GPU_SPAN` probe saw them,
/// summed over the flushes that inference issued.
///
/// This is deployment-like telemetry: the probe keeps chunked submission and
/// costs two timestamps per command buffer, unlike the `VULKAN_EP_STATS`
/// profiler, whose per-dispatch timestamps force one buffer and whose numbers
/// are for attribution, not for timing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuSpan {
    /// Flushes that contributed.
    pub flushes: u32,
    /// Command buffers the probe timed.
    pub command_buffers: u32,
    /// Queue submissions made since the last reset, of any path. Fewer timed
    /// command buffers than submissions means some GPU work was not timed.
    pub submits: u64,
    /// Union of the command buffers' intervals, per flush, summed.
    pub busy_ms: f64,
    /// First begin to last end, per flush, summed. Never below `busy_ms`.
    pub span_ms: f64,
    /// False when a flush lost timestamps (more command buffers than the query
    /// pool holds, or a failed read) or when `submits` exceeds the timed
    /// command buffers: the figures are then lower bounds, not evidence.
    pub complete: bool,
}

/// Accumulates [`GpuSpan`] of the flushes between two takes.
///
/// Separate from the process-wide instance so that the accounting is testable
/// without a device.
pub struct GpuSpanMeter {
    total: Option<GpuSpan>,
    submits_base: u64,
}

impl GpuSpanMeter {
    pub const fn new() -> Self {
        Self {
            total: None,
            submits_base: 0,
        }
    }

    /// Adds one flush. `flush.complete` is false for a flush that lost data.
    pub fn record(&mut self, flush: GpuSpan) {
        let total = self.total.get_or_insert(GpuSpan {
            complete: true,
            ..GpuSpan::default()
        });
        total.flushes += flush.flushes;
        total.command_buffers += flush.command_buffers;
        total.busy_ms += flush.busy_ms;
        total.span_ms += flush.span_ms;
        total.complete &= flush.complete;
    }

    /// Discards what was recorded, so that the next take only covers what runs
    /// after this call. `submits_now` is the process's submission count.
    pub fn reset(&mut self, submits_now: u64) {
        self.total = None;
        self.submits_base = submits_now;
    }

    /// The flushes since the last take or reset, or `None` when the probe saw
    /// none (probe off, no timestamp support, or no GPU work): absence of
    /// evidence is reported as absence, never as a zero.
    pub fn take(&mut self, submits_now: u64) -> Option<GpuSpan> {
        let mut span = self.total.take()?;
        span.submits = submits_now.saturating_sub(self.submits_base);
        span.complete &= u64::from(span.command_buffers) == span.submits;
        self.submits_base = submits_now;
        Some(span)
    }
}

impl Default for GpuSpanMeter {
    fn default() -> Self {
        Self::new()
    }
}

static GPU_SPAN: Mutex<GpuSpanMeter> = Mutex::new(GpuSpanMeter::new());

/// One flush's probe result, from the stream.
pub(crate) fn record_gpu_span(flush: GpuSpan) {
    GPU_SPAN.lock().unwrap().record(flush);
}

/// Starts a measurement window: call before the warm-up of each backend, so
/// that one backend's flushes are never attributed to the next.
pub fn reset_gpu_span() {
    GPU_SPAN
        .lock()
        .unwrap()
        .reset(SUBMITS.load(Ordering::Relaxed));
}

/// Ends the current inference's window and starts the next one.
pub fn take_gpu_span() -> Option<GpuSpan> {
    GPU_SPAN
        .lock()
        .unwrap()
        .take(SUBMITS.load(Ordering::Relaxed))
}

/// Storage-buffer allocations so far (probe hook, see interp's ALLOC_PROBE).
pub fn allocs() -> u64 {
    ALLOCS.load(Ordering::Relaxed)
}

/// Storage-buffer requests the pool served so far: with `allocs`, every
/// buffer a run asked for, whether or not it was recycled.
pub fn pool_hits() -> u64 {
    POOL_HITS.load(Ordering::Relaxed)
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

/// Diagnostic instrumentation this process was started with. A report carries
/// it so that a consumer can check the measurement mode it asked for, instead
/// of trusting that its environment reached the process unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instrumentation {
    /// `VULKAN_EP_STATS`: per-dispatch timestamps, one command buffer.
    pub stats: bool,
    /// `VULKAN_GPU_SPAN`: two timestamps per command buffer.
    pub gpu_span: bool,
    pub node_probe: bool,
    pub alloc_probe: bool,
    pub pool_probe: bool,
}

pub fn instrumentation() -> Instrumentation {
    let present = |name| std::env::var_os(name).is_some();
    Instrumentation {
        stats: enabled(),
        gpu_span: crate::stream::span_probe(),
        node_probe: present("NODE_PROBE"),
        alloc_probe: present("ALLOC_PROBE"),
        pool_probe: present("POOL_PROBE"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flush(command_buffers: u32, busy_ms: f64, span_ms: f64) -> GpuSpan {
        GpuSpan {
            flushes: 1,
            command_buffers,
            busy_ms,
            span_ms,
            complete: true,
            ..GpuSpan::default()
        }
    }

    #[test]
    fn a_window_without_flushes_is_absent_not_zero() {
        let mut meter = GpuSpanMeter::new();
        assert_eq!(meter.take(5), None);
        meter.record(flush(2, 1.0, 1.5));
        assert!(meter.take(2).is_some());
        assert_eq!(meter.take(2), None);
    }

    #[test]
    fn the_flushes_of_one_inference_are_summed_and_the_window_resets() {
        let mut meter = GpuSpanMeter::new();
        meter.reset(10);
        meter.record(flush(2, 1.0, 1.5));
        meter.record(flush(1, 0.5, 0.5));
        let first = meter.take(13).unwrap();
        assert_eq!(first.flushes, 2);
        assert_eq!(first.command_buffers, 3);
        assert_eq!(first.submits, 3);
        assert!((first.busy_ms - 1.5).abs() < 1e-12 && (first.span_ms - 2.0).abs() < 1e-12);
        assert!(first.complete);

        // the next inference starts from nothing, not from the previous total
        meter.record(flush(4, 3.0, 3.0));
        let second = meter.take(17).unwrap();
        assert_eq!(
            (second.flushes, second.command_buffers, second.submits),
            (1, 4, 4)
        );
        assert!((second.busy_ms - 3.0).abs() < 1e-12);
    }

    #[test]
    fn reset_drops_what_an_earlier_backend_recorded() {
        let mut meter = GpuSpanMeter::new();
        meter.record(flush(2, 9.0, 9.0)); // e.g. the EP's last inference
        meter.reset(2);
        assert_eq!(meter.take(2), None);
        meter.record(flush(1, 1.0, 1.0)); // the standalone run
        let span = meter.take(3).unwrap();
        assert_eq!(span.command_buffers, 1);
        assert!((span.busy_ms - 1.0).abs() < 1e-12);
    }

    #[test]
    fn untimed_submissions_or_lost_timestamps_make_the_span_incomplete() {
        let mut meter = GpuSpanMeter::new();
        meter.reset(0);
        meter.record(flush(2, 1.0, 1.0));
        assert!(
            !meter.take(3).unwrap().complete,
            "a submission nobody timed"
        );

        meter.record(GpuSpan {
            complete: false,
            ..flush(2, 1.0, 1.0)
        });
        assert!(
            !meter.take(5).unwrap().complete,
            "a flush that lost timestamps"
        );
        meter.record(flush(1, 1.0, 1.0));
        assert!(
            meter.take(6).unwrap().complete,
            "incompleteness does not leak into the next window"
        );
    }
}
