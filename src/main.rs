use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::Response,
    routing::{get, post},
    Router,
};
use bytemuck::{Pod, Zeroable};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper_util::rt::TokioIo;
use memmap2::Mmap;
use std::cell::RefCell;
use std::fs::File;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use tower::ServiceExt;

// ─── Binary format (must match preprocess.rs) ────────────────────────────────

const SCALE: f32 = 10000.0;
const NULL: u32 = u32::MAX;
const K: usize = 5;
const MAX_PARTITIONS: usize = 256;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Header {
    magic: [u8; 8],
    n_partitions: u32,
    n_nodes: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct PartitionHeader {
    key: u32,
    root: u32,
    count: u32,
    _pad: u32,
    min: [i16; 16],
    max: [i16; 16],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Node {
    vector: [i16; 16],
    radius_sq: i32,
    left: u32,
    right: u32,
    label: u8,
    _pad: [u8; 3],
}

// ─── Fixed responses ──────────────────────────────────────────────────────────

static RESPONSES: [&[u8]; 6] = [
    br#"{"approved":true,"fraud_score":0.0}"#,
    br#"{"approved":true,"fraud_score":0.2}"#,
    br#"{"approved":true,"fraud_score":0.4}"#,
    br#"{"approved":false,"fraud_score":0.6}"#,
    br#"{"approved":false,"fraud_score":0.8}"#,
    br#"{"approved":false,"fraud_score":1.0}"#,
];

// ─── Normalization constants ──────────────────────────────────────────────────

const MAX_AMOUNT: f32 = 10_000.0;
const MAX_INSTALLMENTS: f32 = 12.0;
const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
const MAX_MINUTES: f32 = 1_440.0;
const MAX_KM: f32 = 1_000.0;
const MAX_TX_COUNT_24H: f32 = 20.0;
const MAX_MERCHANT_AVG_AMOUNT: f32 = 10_000.0;

fn mcc_risk(mcc: &str) -> f32 {
    match mcc {
        "5411" => 0.15,
        "5812" => 0.30,
        "5912" => 0.20,
        "5944" => 0.45,
        "7801" => 0.80,
        "7802" => 0.75,
        "7995" => 0.85,
        "4511" => 0.35,
        "5311" => 0.25,
        "5999" => 0.50,
        _ => 0.5,
    }
}

#[inline(always)]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

// ─── API types ────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct AuthRequest {
    transaction: Transaction,
    customer: Customer,
    merchant: Merchant,
    terminal: Terminal,
    last_transaction: Option<LastTransaction>,
}

#[derive(serde::Deserialize)]
struct Transaction {
    amount: f32,
    installments: u32,
    requested_at: String,
}

#[derive(serde::Deserialize)]
struct Customer {
    avg_amount: f32,
    tx_count_24h: u32,
    known_merchants: Vec<String>,
}

#[derive(serde::Deserialize)]
struct Merchant {
    id: String,
    mcc: String,
    avg_amount: f32,
}

#[derive(serde::Deserialize)]
struct Terminal {
    is_online: bool,
    card_present: bool,
    km_from_home: f32,
}

#[derive(serde::Deserialize)]
struct LastTransaction {
    timestamp: String,
    km_from_current: f32,
}

// ─── Vectorization ────────────────────────────────────────────────────────────

fn vectorize(req: &AuthRequest) -> [f32; 14] {
    let t = &req.transaction;
    let c = &req.customer;
    let m = &req.merchant;
    let term = &req.terminal;

    let (hour, dow) = parse_datetime(&t.requested_at);

    let (minutes_since, km_from_last) = if let Some(lt) = &req.last_transaction {
        let minutes = parse_minutes_between(&lt.timestamp, &t.requested_at);
        let km = clamp01(lt.km_from_current / MAX_KM);
        (clamp01(minutes / MAX_MINUTES), km)
    } else {
        (-1.0, -1.0)
    };

    let unknown_merchant = if c.known_merchants.iter().any(|id| id == &m.id) {
        0.0
    } else {
        1.0
    };

    [
        clamp01(t.amount / MAX_AMOUNT),
        clamp01(t.installments as f32 / MAX_INSTALLMENTS),
        clamp01((t.amount / c.avg_amount.max(1e-9)) / AMOUNT_VS_AVG_RATIO),
        hour as f32 / 23.0,
        dow as f32 / 6.0,
        minutes_since,
        km_from_last,
        clamp01(term.km_from_home / MAX_KM),
        clamp01(c.tx_count_24h as f32 / MAX_TX_COUNT_24H),
        if term.is_online { 1.0 } else { 0.0 },
        if term.card_present { 1.0 } else { 0.0 },
        unknown_merchant,
        mcc_risk(&m.mcc),
        clamp01(m.avg_amount / MAX_MERCHANT_AVG_AMOUNT),
    ]
}

fn parse_datetime(s: &str) -> (u32, u32) {
    let bytes = s.as_bytes();
    if bytes.len() < 19 {
        return (0, 0);
    }
    let hour = parse_u32(&bytes[11..13]);
    let year = parse_u32(&bytes[0..4]);
    let month = parse_u32(&bytes[5..7]);
    let day = parse_u32(&bytes[8..10]);
    let dow = day_of_week(year, month, day);
    (hour, dow)
}

fn parse_u32(bytes: &[u8]) -> u32 {
    let mut v = 0u32;
    for &b in bytes {
        if b >= b'0' && b <= b'9' {
            v = v * 10 + (b - b'0') as u32;
        }
    }
    v
}

fn day_of_week(year: u32, month: u32, day: u32) -> u32 {
    let y = if month < 3 { year - 1 } else { year };
    let m = month as usize;
    static T: [u32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let dow = (y + y / 4 - y / 100 + y / 400 + T[m - 1] + day) % 7;
    (dow + 6) % 7
}

fn parse_minutes_between(prev: &str, curr: &str) -> f32 {
    let prev_secs = parse_unix_secs(prev) as i64;
    let curr_secs = parse_unix_secs(curr) as i64;
    let diff_secs = (curr_secs - prev_secs).abs();
    diff_secs as f32 / 60.0
}

fn parse_unix_secs(s: &str) -> u64 {
    let bytes = s.as_bytes();
    if bytes.len() < 19 {
        return 0;
    }
    let year = parse_u32(&bytes[0..4]) as u64;
    let month = parse_u32(&bytes[5..7]) as u64;
    let day = parse_u32(&bytes[8..10]) as u64;
    let hour = parse_u32(&bytes[11..13]) as u64;
    let min = parse_u32(&bytes[14..16]) as u64;
    let sec = parse_u32(&bytes[17..19]) as u64;

    let days = days_since_epoch(year, month, day);
    days * 86400 + hour * 3600 + min * 60 + sec
}

fn days_since_epoch(year: u64, month: u64, day: u64) -> u64 {
    let y = if month <= 2 { year - 1 } else { year };
    let m = if month <= 2 { month + 9 } else { month - 3 };
    let era = y / 400;
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

// ─── Partition key (must match preprocess.rs) ─────────────────────────────────

fn compute_partition_key(v: &[f32; 14]) -> u8 {
    let mut k: u8 = 0;
    if v[5] >= 0.0 { k |= 1 << 0; }
    if v[9]  > 0.5 { k |= 1 << 1; }
    if v[10] > 0.5 { k |= 1 << 2; }
    if v[11] > 0.5 { k |= 1 << 3; }

    let mcc = v[12];
    let bucket: u8 = if mcc <= 0.2 { 0 }
                     else if mcc <= 0.5 { 1 }
                     else if mcc <= 0.8 { 2 }
                     else { 3 };
    k |= bucket << 4;

    if v[2] > 0.5 { k |= 1 << 6; }
    if v[8] > 0.5 { k |= 1 << 7; }
    k
}

// ─── Quantization ─────────────────────────────────────────────────────────────

#[inline(always)]
fn quantize(v: f32) -> i16 {
    let s = (v * SCALE).round();
    if s >= 32767.0 {
        32767
    } else if s <= -32768.0 {
        -32768
    } else {
        s as i16
    }
}

fn quantize_vec(v: &[f32; 14]) -> [i16; 16] {
    let mut out = [0i16; 16];
    for i in 0..14 {
        out[i] = quantize(v[i]);
    }
    out
}

// ─── Distance (SSE2 i16 via _mm_madd_epi16, 128-bit avoids AVX-256 downclock) ─

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn dist_sq_i16_sse2(a: &[i16; 16], b: &[i16; 16]) -> i32 {
    use std::arch::x86_64::*;
    // Process two 128-bit halves (8 i16 each = 16 i16 total)
    let a_lo = _mm_loadu_si128(a.as_ptr() as *const __m128i);
    let b_lo = _mm_loadu_si128(b.as_ptr() as *const __m128i);
    let a_hi = _mm_loadu_si128(a.as_ptr().add(8) as *const __m128i);
    let b_hi = _mm_loadu_si128(b.as_ptr().add(8) as *const __m128i);

    let d_lo = _mm_sub_epi16(a_lo, b_lo);
    let d_hi = _mm_sub_epi16(a_hi, b_hi);

    let m_lo = _mm_madd_epi16(d_lo, d_lo); // 4 i32
    let m_hi = _mm_madd_epi16(d_hi, d_hi);

    let sum = _mm_add_epi32(m_lo, m_hi);
    let s2 = _mm_add_epi32(sum, _mm_shuffle_epi32::<0b_01_00_11_10>(sum));
    let s1 = _mm_add_epi32(s2, _mm_shuffle_epi32::<0b_00_00_00_01>(s2));
    _mm_cvtsi128_si32(s1)
}

#[inline(always)]
fn dist_sq_i16(a: &[i16; 16], b: &[i16; 16]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { dist_sq_i16_sse2(a, b) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut sum: i32 = 0;
        for i in 0..14 {
            let d = a[i] as i32 - b[i] as i32;
            sum += d * d;
        }
        sum
    }
}

// ─── Lower-bound distance² to bounding box ────────────────────────────────────

#[inline(always)]
fn lb_box_sq(q: &[i16; 16], min: &[i16; 16], max: &[i16; 16]) -> i32 {
    let mut sum: i64 = 0;
    for i in 0..14 {
        let qi = q[i] as i32;
        let mi = min[i] as i32;
        let mx = max[i] as i32;
        let d = if qi < mi {
            mi - qi
        } else if qi > mx {
            qi - mx
        } else {
            0
        };
        sum += (d * d) as i64;
    }
    if sum > i32::MAX as i64 { i32::MAX } else { sum as i32 }
}

// ─── KNN search ───────────────────────────────────────────────────────────────

struct AppState {
    #[allow(dead_code)]
    mmap: Mmap,
    partitions_ptr: usize,
    partitions_len: usize,
    nodes_ptr: usize,
    nodes_len: usize,
    // key → index into partitions[], or u16::MAX if absent
    key_to_idx: [u16; 256],
}

unsafe impl Send for AppState {}
unsafe impl Sync for AppState {}

impl AppState {
    fn new(mmap: Mmap) -> Self {
        assert!(mmap.len() >= std::mem::size_of::<Header>(), "file too small");
        let header: &Header = bytemuck::from_bytes(&mmap[..std::mem::size_of::<Header>()]);
        assert_eq!(&header.magic, b"RNSPSCT1", "bad magic");

        let n_partitions = header.n_partitions as usize;
        let n_nodes = header.n_nodes as usize;

        let part_offset = std::mem::size_of::<Header>();
        let part_size = n_partitions * std::mem::size_of::<PartitionHeader>();
        let nodes_offset = part_offset + part_size;
        let nodes_size = n_nodes * std::mem::size_of::<Node>();
        assert_eq!(mmap.len(), nodes_offset + nodes_size, "file size mismatch");

        let partitions: &[PartitionHeader] =
            bytemuck::cast_slice(&mmap[part_offset..part_offset + part_size]);
        let nodes: &[Node] =
            bytemuck::cast_slice(&mmap[nodes_offset..nodes_offset + nodes_size]);

        let partitions_ptr = partitions.as_ptr() as usize;
        let partitions_len = partitions.len();
        let nodes_ptr = nodes.as_ptr() as usize;
        let nodes_len = nodes.len();

        // Build O(1) lookup table
        let mut key_to_idx = [u16::MAX; 256];
        for (i, p) in partitions.iter().enumerate() {
            key_to_idx[p.key as usize & 0xff] = i as u16;
        }

        eprintln!(
            "Loaded: {} partitions, {} nodes ({:.1} MB)",
            n_partitions,
            n_nodes,
            mmap.len() as f64 / 1_000_000.0,
        );

        Self { mmap, partitions_ptr, partitions_len, nodes_ptr, nodes_len, key_to_idx }
    }

    #[inline(always)]
    fn partitions(&self) -> &[PartitionHeader] {
        unsafe { std::slice::from_raw_parts(self.partitions_ptr as *const PartitionHeader, self.partitions_len) }
    }

    #[inline(always)]
    fn nodes(&self) -> &[Node] {
        unsafe { std::slice::from_raw_parts(self.nodes_ptr as *const Node, self.nodes_len) }
    }
}

thread_local! {
    static STACK_BUF: RefCell<Vec<u32>> = RefCell::new(Vec::with_capacity(256));
}

fn search_tree(
    nodes: &[Node],
    root: u32,
    query: &[i16; 16],
    best: &mut [(i32, u8); K],
    worst_best: &mut i32,
) {
    STACK_BUF.with(|buf| {
        let mut stack = buf.borrow_mut();
        stack.clear();
        stack.push(root);

        while let Some(idx) = stack.pop() {
            if idx == NULL { continue; }
            let node = &nodes[idx as usize];
            let d = dist_sq_i16(query, &node.vector);

            if d < *worst_best {
                let mut max_pos = 0;
                for i in 1..K {
                    if best[i].0 > best[max_pos].0 { max_pos = i; }
                }
                best[max_pos] = (d, node.label);
                let mut w = best[0].0;
                for i in 1..K {
                    if best[i].0 > w { w = best[i].0; }
                }
                *worst_best = w;
            }

            // VP-tree pruning via triangle inequality. Compare squared dists
            // (cheaper than sqrt on worst_best — matches the original f32 logic).
            let d_f = (d as f32).sqrt();
            let r_f = (node.radius_sq as f32).sqrt();
            let dl = (d_f - r_f).max(0.0);
            let dr = (r_f - d_f).max(0.0);
            let dl_sq = (dl * dl) as i32;
            let dr_sq = (dr * dr) as i32;
            let can_left  = node.left  != NULL && dl_sq < *worst_best;
            let can_right = node.right != NULL && dr_sq < *worst_best;

            if can_left && can_right {
                if d <= node.radius_sq {
                    stack.push(node.right);
                    stack.push(node.left);
                } else {
                    stack.push(node.left);
                    stack.push(node.right);
                }
            } else {
                if can_left  { stack.push(node.left); }
                if can_right { stack.push(node.right); }
            }
        }
    });
}

fn knn_search(state: &AppState, query: &[i16; 16], key: u8) -> usize {
    let partitions = state.partitions();
    let nodes = state.nodes();

    let mut best = [(i32::MAX, 0u8); K];
    let mut worst_best = i32::MAX;

    // 1) Primary partition first (key-first) — O(1) lookup
    let primary_idx_raw = state.key_to_idx[key as usize];
    let primary_idx: Option<usize> = if primary_idx_raw == u16::MAX {
        None
    } else {
        Some(primary_idx_raw as usize)
    };

    if let Some(idx) = primary_idx {
        search_tree(nodes, partitions[idx].root, query, &mut best, &mut worst_best);
    }

    // 2) Other partitions, ordered by lb²
    let mut others: [(i32, u32); MAX_PARTITIONS] = [(i32::MAX, NULL); MAX_PARTITIONS];
    let mut count = 0;
    for (i, p) in partitions.iter().enumerate() {
        if Some(i) == primary_idx { continue; }
        others[count] = (lb_box_sq(query, &p.min, &p.max), p.root);
        count += 1;
    }
    others[..count].sort_unstable_by_key(|(lb, _)| *lb);

    for &(lb, root) in &others[..count] {
        if lb >= worst_best { break; }
        search_tree(nodes, root, query, &mut best, &mut worst_best);
    }

    let sum: usize = best.iter().map(|(_, l)| *l as usize).sum();
    sum.min(5)
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

#[inline(always)]
fn json_response(data: &'static [u8]) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(data))
        .unwrap()
}

async fn ready_handler() -> impl axum::response::IntoResponse {
    StatusCode::OK
}

async fn fraud_score_handler(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    let req: AuthRequest = match sonic_rs::from_slice(&body) {
        Ok(r) => r,
        Err(_) => return json_response(RESPONSES[0]),
    };

    if state.nodes_len == 0 {
        return json_response(RESPONSES[0]);
    }

    let v = vectorize(&req);
    let key = compute_partition_key(&v);
    let query = quantize_vec(&v);
    let fraud_count = knn_search(&state, &query, key);
    json_response(RESPONSES[fraud_count])
}

// ─── Warmup ───────────────────────────────────────────────────────────────────

fn warmup_mmap(mmap: &Mmap) {
    let page_size = 4096usize;
    let len = mmap.len();
    let mut sum = 0u8;
    let mut i = 0;
    while i < len {
        sum = sum.wrapping_add(mmap[i]);
        i += page_size;
    }
    let _ = std::hint::black_box(sum);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let vptree_path = std::env::var("VPTREE_PATH").unwrap_or_else(|_| "vptree.bin".to_string());

    let file = File::open(&vptree_path).expect("cannot open vptree.bin");
    let mmap = unsafe { Mmap::map(&file).expect("mmap failed") };

    eprintln!("Warming up mmap ({} MB)...", mmap.len() / 1_000_000);
    warmup_mmap(&mmap);

    let state = Arc::new(AppState::new(mmap));

    let app = Router::new()
        .route("/ready", get(ready_handler))
        .route("/fraud-score", post(fraud_score_handler))
        .with_state(state);

    // TCP :8080 — apenas para healthcheck do docker-compose
    let tcp_listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .expect("tcp bind failed");
    let tcp_app = app.clone();
    tokio::spawn(async move {
        axum::serve(tcp_listener, tcp_app).await.expect("tcp serve failed");
    });

    // Unix socket — tráfego real do HAProxy
    let sock_path = std::env::var("SOCKET_PATH").unwrap_or_else(|_| "/tmp/api.sock".to_string());
    let _ = std::fs::remove_file(&sock_path);
    let unix_listener = tokio::net::UnixListener::bind(&sock_path).expect("unix bind failed");
    std::fs::set_permissions(&sock_path, std::fs::Permissions::from_mode(0o777))
        .expect("set socket permissions failed");
    eprintln!("Listening on {sock_path} (Unix) and :8080 (TCP health)");

    loop {
        let (stream, _) = unix_listener.accept().await.expect("unix accept failed");
        let io = TokioIo::new(stream);
        let app = app.clone();
        tokio::spawn(async move {
            let svc = hyper::service::service_fn(move |req: hyper::Request<Incoming>| {
                app.clone().oneshot(req.map(Body::new))
            });
            http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, svc)
                .await
                .ok();
        });
    }
}
