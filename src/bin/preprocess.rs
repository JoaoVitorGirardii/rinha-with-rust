use bytemuck::{Pod, Zeroable};
use sonic_rs::{JsonContainerTrait, JsonValueTrait};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, Read, Write};

// ─── Binary format ────────────────────────────────────────────────────────────
//
// File layout:
//   [Header 16B] [PartitionHeader × n_partitions, 80B each] [Node × n_nodes, 48B each]
//
// Vectors are [i16; 16] = 14 real dims + 2 zero-padding for AVX2-friendly 32B loads.

const SCALE: f32 = 10000.0;
const NULL: u32 = u32::MAX;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Header {
    magic: [u8; 8],     // "RNSPSCT1"
    n_partitions: u32,
    n_nodes: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct PartitionHeader {
    key: u32,           // partition key (8-bit value stored as u32 for alignment)
    root: u32,          // global offset into Nodes array; NULL if empty
    count: u32,         // number of nodes in this partition
    _pad: u32,
    min: [i16; 16],     // bounding box min (14 dims + 2 zeros)
    max: [i16; 16],     // bounding box max
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Node {
    vector: [i16; 16],  // 14 features + 2 zeros (AVX2 alignment)
    radius_sq: i32,     // squared median distance (i16 scale)
    left: u32,
    right: u32,
    label: u8,
    _pad: [u8; 3],
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

// ─── Partition key (8 bits) ───────────────────────────────────────────────────
//
// bit 0: has last_transaction (v[5] >= 0, since -1 means absent)
// bit 1: is_online            (v[9]  > 0.5)
// bit 2: card_present         (v[10] > 0.5)
// bit 3: unknown_merchant     (v[11] > 0.5)
// bits 4-5: mcc bucket (4 levels)
// bit 6: high_value           (v[2]  > 0.5  ≡ amount/avg > 5)
// bit 7: frequent_tx          (v[8]  > 0.5  ≡ tx_count_24h > 10)

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

// ─── Distance (i32, since acumulação cabe — pior caso 2e9 < i32::MAX) ─────────

#[inline(always)]
fn dist_sq_i16(a: &[i16; 16], b: &[i16; 16]) -> i32 {
    let mut sum: i32 = 0;
    for i in 0..14 {
        let d = a[i] as i32 - b[i] as i32;
        sum += d * d;
    }
    sum
}

// ─── VP-Tree build (per partition) ────────────────────────────────────────────

fn build_vptree(points: &mut Vec<([i16; 16], u8)>) -> Vec<Node> {
    let n = points.len();
    let mut nodes: Vec<Node> = Vec::with_capacity(n);

    enum Patch {
        Left(usize),
        Right(usize),
        Root,
    }
    let mut stack: Vec<(usize, usize, Patch)> = vec![(0, n, Patch::Root)];
    let mut root_idx: u32 = 0;

    while let Some((start, end, patch)) = stack.pop() {
        let len = end - start;
        if len == 0 {
            match patch {
                Patch::Left(p) => nodes[p].left = NULL,
                Patch::Right(p) => nodes[p].right = NULL,
                Patch::Root => {}
            }
            continue;
        }

        // vantage point: sample up to 15 candidates, pick highest avg dispersion
        let sample_size = 15.min(len);
        let step = if len > sample_size { len / sample_size } else { 1 };
        let vp_offset = {
            let mut best_offset = 0usize;
            let mut best_disp: i64 = -1;
            for s in 0..sample_size {
                let idx = start + (s * step).min(len - 1);
                let mut disp: i64 = 0;
                let mut count: i64 = 0;
                for s2 in 0..sample_size {
                    let idx2 = start + (s2 * step).min(len - 1);
                    if idx != idx2 {
                        disp += dist_sq_i16(&points[idx].0, &points[idx2].0) as i64;
                        count += 1;
                    }
                }
                if count > 0 {
                    disp /= count;
                }
                if disp > best_disp {
                    best_disp = disp;
                    best_offset = idx - start;
                }
            }
            best_offset
        };

        points.swap(start, start + vp_offset);
        let vp = points[start].0;
        let label = points[start].1;

        let rest = &mut points[start + 1..end];
        let mut dists: Vec<i32> = rest.iter().map(|p| dist_sq_i16(&p.0, &vp)).collect();

        let radius_sq = if dists.is_empty() {
            0
        } else {
            let mid = dists.len() / 2;
            dists.select_nth_unstable(mid);
            dists[mid]
        };

        let rest_len = rest.len();
        let dists2: Vec<i32> = rest.iter().map(|p| dist_sq_i16(&p.0, &vp)).collect();
        let mut left_indices: Vec<usize> = Vec::new();
        let mut right_indices: Vec<usize> = Vec::new();
        for i in 0..rest_len {
            if dists2[i] <= radius_sq {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }
        let tmp: Vec<([i16; 16], u8)> = rest.to_vec();
        let split = left_indices.len();
        for (dst, &src) in left_indices.iter().enumerate() {
            rest[dst] = tmp[src];
        }
        for (dst, &src) in right_indices.iter().enumerate() {
            rest[split + dst] = tmp[src];
        }

        let node_idx = nodes.len();
        nodes.push(Node {
            vector: vp,
            radius_sq,
            left: NULL,
            right: NULL,
            label,
            _pad: [0; 3],
        });

        match patch {
            Patch::Left(p) => nodes[p].left = node_idx as u32,
            Patch::Right(p) => nodes[p].right = node_idx as u32,
            Patch::Root => root_idx = node_idx as u32,
        }

        let left_end = start + 1 + split;
        let right_end = end;
        let right_start = left_end;

        if right_start < right_end {
            stack.push((right_start, right_end, Patch::Right(node_idx)));
        }
        if start + 1 < left_end {
            stack.push((start + 1, left_end, Patch::Left(node_idx)));
        }
    }

    // BFS reorder
    let mut bfs: Vec<Node> = Vec::with_capacity(nodes.len());
    let mut old_to_new: Vec<u32> = vec![NULL; nodes.len()];
    let mut queue: VecDeque<u32> = VecDeque::new();
    queue.push_back(root_idx);

    while let Some(old_idx) = queue.pop_front() {
        let new_idx = bfs.len() as u32;
        old_to_new[old_idx as usize] = new_idx;
        bfs.push(nodes[old_idx as usize]);

        let node = &nodes[old_idx as usize];
        if node.left != NULL {
            queue.push_back(node.left);
        }
        if node.right != NULL {
            queue.push_back(node.right);
        }
    }

    for node in &mut bfs {
        if node.left != NULL {
            node.left = old_to_new[node.left as usize];
        }
        if node.right != NULL {
            node.right = old_to_new[node.right as usize];
        }
    }

    bfs
}

// ─── Bounding box ─────────────────────────────────────────────────────────────

fn compute_bbox(points: &[([i16; 16], u8)]) -> ([i16; 16], [i16; 16]) {
    let mut min = [i16::MAX; 16];
    let mut max = [i16::MIN; 16];
    for (v, _) in points {
        for i in 0..14 {
            if v[i] < min[i] { min[i] = v[i]; }
            if v[i] > max[i] { max[i] = v[i]; }
        }
    }
    // padding lanes stay at 0 (we never query non-zero there)
    for i in 14..16 {
        min[i] = 0;
        max[i] = 0;
    }
    (min, max)
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let gz_path = std::env::args().nth(1).unwrap_or_else(|| "references.json.gz".to_string());
    let out_path = std::env::args().nth(2).unwrap_or_else(|| "vptree.bin".to_string());

    eprintln!("Reading {gz_path}...");
    let file = File::open(&gz_path).expect("cannot open references.json.gz");
    let decoder = flate2::read::GzDecoder::new(BufReader::new(file));
    let mut reader = BufReader::new(decoder);
    let mut json_bytes = Vec::new();
    reader.read_to_end(&mut json_bytes).expect("read error");

    eprintln!("Parsing JSON ({} MB)...", json_bytes.len() / 1_000_000);
    let root: sonic_rs::Value = sonic_rs::from_slice(&json_bytes).expect("json parse error");
    let arr = root.as_array().expect("expected array");

    eprintln!("Loaded {} references, partitioning...", arr.len());

    // Group references by partition key
    let mut groups: Vec<Vec<([i16; 16], u8)>> = (0..256).map(|_| Vec::new()).collect();

    for item in arr.iter() {
        let vec_arr = item["vector"].as_array().expect("vector field");
        let mut v = [0.0f32; 14];
        for (i, x) in vec_arr.iter().enumerate() {
            if i >= 14 { break; }
            v[i] = x.as_f64().unwrap_or(0.0) as f32;
        }
        let label_str = item["label"].as_str().unwrap_or("legit");
        let label: u8 = if label_str == "fraud" { 1 } else { 0 };

        let key = compute_partition_key(&v) as usize;
        let qv = quantize_vec(&v);
        groups[key].push((qv, label));
    }

    let n_non_empty = groups.iter().filter(|g| !g.is_empty()).count();
    eprintln!("{} non-empty partitions out of 256.", n_non_empty);

    // Print distribution (helps debug skew)
    let mut sizes: Vec<(usize, usize)> = groups.iter().enumerate()
        .filter(|(_, g)| !g.is_empty())
        .map(|(i, g)| (i, g.len()))
        .collect();
    sizes.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    eprintln!("Top partitions: ");
    for (k, n) in sizes.iter().take(10) {
        eprintln!("  key={k:3} count={n}");
    }
    eprintln!("Smallest: ");
    for (k, n) in sizes.iter().rev().take(5) {
        eprintln!("  key={k:3} count={n}");
    }

    // Build VP-tree per partition, accumulate into global node list
    let mut all_nodes: Vec<Node> = Vec::new();
    let mut partitions: Vec<PartitionHeader> = Vec::new();

    for (key, group) in groups.iter_mut().enumerate() {
        if group.is_empty() {
            continue;
        }
        let (min, max) = compute_bbox(group);
        let local_nodes = build_vptree(group);
        let base = all_nodes.len() as u32;
        let count = local_nodes.len() as u32;

        // Rewrite child offsets to global indices
        for n in &local_nodes {
            let mut g = *n;
            if g.left != NULL { g.left += base; }
            if g.right != NULL { g.right += base; }
            all_nodes.push(g);
        }

        partitions.push(PartitionHeader {
            key: key as u32,
            root: base, // root is at local index 0 after BFS reorder
            count,
            _pad: 0,
            min,
            max,
        });
    }

    let n_partitions = partitions.len() as u32;
    let n_nodes = all_nodes.len() as u32;
    eprintln!("Built {} partitions, {} total nodes.", n_partitions, n_nodes);

    let header = Header {
        magic: *b"RNSPSCT1",
        n_partitions,
        n_nodes,
    };

    let mut out = File::create(&out_path).expect("cannot create output file");
    out.write_all(bytemuck::bytes_of(&header)).expect("write header");
    out.write_all(bytemuck::cast_slice(&partitions)).expect("write partitions");
    out.write_all(bytemuck::cast_slice(&all_nodes)).expect("write nodes");

    let total_bytes = std::mem::size_of::<Header>()
        + n_partitions as usize * std::mem::size_of::<PartitionHeader>()
        + n_nodes as usize * std::mem::size_of::<Node>();
    eprintln!("Done. File size: {:.1} MB", total_bytes as f64 / 1_000_000.0);
}
