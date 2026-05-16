use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::Response,
    routing::{get, post},
    Router,
};
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use std::cell::RefCell;
use std::fs::File;
use std::sync::Arc;

// ─── VP-Tree node (must match preprocess.rs exactly) ─────────────────────────

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Node {
    vector: [f32; 14],
    radius: f32,
    left: u32,
    right: u32,
    label: u8,
    _pad: [u8; 3],
}

const NULL: u32 = u32::MAX;
const K: usize = 5;

// fraud_count ∈ {0,1,2,3,4,5} → score ∈ {0.0,0.2,0.4,0.6,0.8,1.0}
// approved = score < 0.6 → fraud_count < 3
static RESPONSES: [&[u8]; 6] = [
    br#"{"approved":true,"fraud_score":0.0}"#,
    br#"{"approved":true,"fraud_score":0.2}"#,
    br#"{"approved":true,"fraud_score":0.4}"#,
    br#"{"approved":false,"fraud_score":0.6}"#,
    br#"{"approved":false,"fraud_score":0.8}"#,
    br#"{"approved":false,"fraud_score":1.0}"#,
];

// ─── Normalization constants (baked in — from normalization.json) ─────────────

const MAX_AMOUNT: f32 = 10_000.0;
const MAX_INSTALLMENTS: f32 = 12.0;
const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
const MAX_MINUTES: f32 = 1_440.0;
const MAX_KM: f32 = 1_000.0;
const MAX_TX_COUNT_24H: f32 = 20.0;
const MAX_MERCHANT_AVG_AMOUNT: f32 = 10_000.0;

// MCC risk table (baked in — from mcc_risk.json)
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

// Returns 0=Mon..6=Sun (ISO weekday - 1)
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

// ─── KNN search on VP-Tree ────────────────────────────────────────────────────

#[inline(always)]
fn dist_sq(a: &[f32; 14], b: &[f32; 14]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..14 {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

struct AppState {
    #[allow(dead_code)]
    nodes: Mmap,
    node_ptr: usize,
    node_len: usize,
}

unsafe impl Send for AppState {}
unsafe impl Sync for AppState {}

impl AppState {
    fn new(nodes: Mmap) -> Self {
        let slice: &[Node] = bytemuck::cast_slice(&nodes[..]);
        let node_ptr = slice.as_ptr() as usize;
        let node_len = slice.len();
        Self { nodes, node_ptr, node_len }
    }

    fn nodes_slice(&self) -> &[Node] {
        unsafe { std::slice::from_raw_parts(self.node_ptr as *const Node, self.node_len) }
    }
}

thread_local! {
    static STACK_BUF: RefCell<Vec<u32>> = RefCell::new(Vec::with_capacity(128));
}

fn knn_search(nodes: &[Node], query: &[f32; 14]) -> usize {
    let mut best = [(f32::MAX, 0u8); K];
    let mut worst_best = f32::MAX;

    STACK_BUF.with(|buf| {
        let mut stack = buf.borrow_mut();
        stack.clear();
        stack.push(0);

        while let Some(idx) = stack.pop() {
            if idx == NULL { continue; }
            let node = &nodes[idx as usize];
            let d = dist_sq(query, &node.vector);

            if d < worst_best {
                let mut max_pos = 0;
                for i in 1..K {
                    if best[i].0 > best[max_pos].0 { max_pos = i; }
                }
                best[max_pos] = (d, node.label);
                worst_best = best.iter().map(|x| x.0).fold(f32::MIN, f32::max);
            }

            let sqrt_d = d.sqrt();
            let r = node.radius;
            let dl = (sqrt_d - r).max(0.0);
            let dr = (r - sqrt_d).max(0.0);
            let can_left  = node.left  != NULL && dl * dl < worst_best;
            let can_right = node.right != NULL && dr * dr < worst_best;

            if can_left && can_right {
                if sqrt_d <= r {
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

    best.iter().map(|&(_, label)| label as usize).sum()
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

    if state.node_len == 0 {
        return json_response(RESPONSES[0]);
    }

    let query = vectorize(&req);
    let fraud_count = knn_search(state.nodes_slice(), &query);
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

    let node_count = mmap.len() / std::mem::size_of::<Node>();
    eprintln!("VP-Tree loaded: {node_count} nodes. Ready.");

    let state = Arc::new(AppState::new(mmap));

    let app = Router::new()
        .route("/ready", get(ready_handler))
        .route("/fraud-score", post(fraud_score_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .expect("bind failed");

    axum::serve(listener, app).await.expect("serve failed");
}
