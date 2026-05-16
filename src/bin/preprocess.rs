use bytemuck::{Pod, Zeroable};
use sonic_rs::{JsonContainerTrait, JsonValueTrait};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, Read, Write};

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

fn dist_sq(a: &[f32; 14], b: &[f32; 14]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..14 {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

fn build_vptree(points: &mut Vec<([f32; 14], u8)>) -> Vec<Node> {
    let n = points.len();
    // arena indexed by insertion order — we'll BFS-reorder at the end
    let mut nodes: Vec<Node> = Vec::with_capacity(n);

    // stack: (slice_start, slice_end, parent_idx, is_right_child)
    // We build recursively via explicit stack to avoid stack overflow on 3M nodes
    // Each entry: (start, end) in `points`, and where to patch the parent pointer
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

        // choose vantage point: sample up to 15 candidates, pick max avg-dispersion
        let sample_size = 15.min(len);
        let step = if len > sample_size { len / sample_size } else { 1 };
        let vp_offset = {
            let mut best_offset = 0usize;
            let mut best_disp = -1.0f32;
            for s in 0..sample_size {
                let idx = start + (s * step).min(len - 1);
                // compute avg distance from this candidate to other samples
                let mut disp = 0.0f32;
                let mut count = 0usize;
                for s2 in 0..sample_size {
                    let idx2 = start + (s2 * step).min(len - 1);
                    if idx != idx2 {
                        disp += dist_sq(&points[idx].0, &points[idx2].0);
                        count += 1;
                    }
                }
                if count > 0 {
                    disp /= count as f32;
                }
                if disp > best_disp {
                    best_disp = disp;
                    best_offset = idx - start;
                }
            }
            best_offset
        };

        // swap vantage point to front
        points.swap(start, start + vp_offset);
        let vp = points[start].0;
        let label = points[start].1;

        // compute distances of remaining points to vp
        let rest = &mut points[start + 1..end];
        let mut dists: Vec<f32> = rest.iter().map(|p| dist_sq(&p.0, &vp)).collect();

        // median distance = radius
        let radius = if dists.is_empty() {
            0.0
        } else {
            let mid = dists.len() / 2;
            // partial sort to find median
            dists.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            dists[mid]
        };

        // partition rest around median (stable enough via index)
        // re-sort rest so that dist<=radius goes left, dist>radius goes right
        let rest_len = rest.len();
        // compute distances again (we mutated dists via select_nth)
        let dists2: Vec<f32> = rest.iter().map(|p| dist_sq(&p.0, &vp)).collect();
        // stable partition: collect indices
        let mut left_indices: Vec<usize> = Vec::new();
        let mut right_indices: Vec<usize> = Vec::new();
        for i in 0..rest_len {
            if dists2[i] <= radius {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }
        // reorder rest in-place: left then right
        let tmp: Vec<([f32; 14], u8)> = rest.to_vec();
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
            radius,
            left: NULL,
            right: NULL,
            label,
            _pad: [0; 3],
        });

        // patch parent
        match patch {
            Patch::Left(p) => nodes[p].left = node_idx as u32,
            Patch::Right(p) => nodes[p].right = node_idx as u32,
            Patch::Root => root_idx = node_idx as u32,
        }

        let left_end = start + 1 + split;
        let right_end = end;
        let right_start = left_end;

        // push right first so left is processed first (preserves approximate BFS ordering benefit)
        if right_start < right_end {
            stack.push((right_start, right_end, Patch::Right(node_idx)));
        } else {
            nodes[node_idx].right = NULL;
        }
        if start + 1 < left_end {
            stack.push((start + 1, left_end, Patch::Left(node_idx)));
        } else {
            nodes[node_idx].left = NULL;
        }
    }

    // BFS reorder: traverse nodes starting from root_idx, emit in BFS order
    let mut bfs: Vec<Node> = Vec::with_capacity(nodes.len());
    // mapping: old_idx -> new_idx
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

    // fix up pointers using old_to_new
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

    // Parse with sonic-rs
    let root: sonic_rs::Value = sonic_rs::from_slice(&json_bytes).expect("json parse error");
    let arr = root.as_array().expect("expected array");

    eprintln!("Loaded {} references, building VP-Tree...", arr.len());

    let mut points: Vec<([f32; 14], u8)> = arr
        .iter()
        .map(|item| {
            let vec_arr = item["vector"].as_array().expect("vector field");
            let mut v = [0.0f32; 14];
            for (i, x) in vec_arr.iter().enumerate() {
                v[i] = x.as_f64().unwrap_or(0.0) as f32;
            }
            let label_str = item["label"].as_str().unwrap_or("legit");
            let label: u8 = if label_str == "fraud" { 1 } else { 0 };
            (v, label)
        })
        .collect();

    let nodes = build_vptree(&mut points);

    eprintln!("VP-Tree built: {} nodes, writing to {out_path}...", nodes.len());

    let bytes: &[u8] = bytemuck::cast_slice(&nodes);
    let mut out = File::create(&out_path).expect("cannot create output file");
    out.write_all(bytes).expect("write error");

    eprintln!(
        "Done. File size: {:.1} MB",
        bytes.len() as f64 / 1_000_000.0
    );
}
