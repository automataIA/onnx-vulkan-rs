//! **Convex** partitioning of supported nodes into fusable groups.
//!
//! A backend can fuse a group only if it is *convex*: no non-member node
//! can exist on a path between two members (otherwise that node remains
//! trapped outside the block). Grouping supported adjacent nodes
//! into maximal convex blocks is what truly reduces CPU↔GPU boundaries:
//! one block = 1 upload / 1 submit / 1 download instead of per node.
//!
//! However, single group convexity is insufficient: contracting **all**
//! groups together, two convex blocks might form a cycle in the
//! quotient graph if directions pass through different external nodes
//! (`A → x → B` and `B → y → A`). ORT rejects the model with "the graph is not
//! acyclic". The correct condition is therefore on the **quotient** graph.
//! cluster must lie both downstream and upstream of the block.
//!
//! Method: precompute the **ancestor** and **descendant** sets of every node on
//! the full DAG (Kahn topological sort, robust to `Graph_GetNodes` ordering).
//! Then run greedy union-find along edges between supported nodes: a merge is
//! accepted only if no external cluster appears simultaneously among the
//! descendants and among the ancestors of the resulting set. The node-only test
//! is the special case where that cluster is a singleton. Ancestors/descendants
//! are static (a property of the original graph) and clusters only grow, so the
//! multi-pass greedy converges.

/// Bitset over `u64` (one word every 64 nodes).
type Bits = Vec<u64>;

fn words(n: usize) -> usize {
    n.div_ceil(64)
}

fn set_bit(b: &mut Bits, i: usize) {
    b[i / 64] |= 1u64 << (i % 64);
}

fn or_into(dst: &mut Bits, src: &Bits) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d |= *s;
    }
}

/// Indices of set bits in `b & !mask`, appended to `out`.
fn set_bits_outside(b: &Bits, mask: &Bits, out: &mut Vec<usize>) {
    out.clear();
    for (k, (&word, &m)) in b.iter().zip(mask.iter()).enumerate() {
        let mut rest = word & !m;
        while rest != 0 {
            let bit = rest.trailing_zeros() as usize;
            out.push(k * 64 + bit);
            rest &= rest - 1;
        }
    }
}

/// Union-find with path compression.
struct Uf {
    parent: Vec<usize>,
}

impl Uf {
    fn new(n: usize) -> Self {
        Uf {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.parent[r] != r {
            r = self.parent[r];
        }
        // path compression
        let mut c = x;
        while self.parent[c] != r {
            let n = self.parent[c];
            self.parent[c] = r;
            c = n;
        }
        r
    }
}

/// Groups supported nodes into maximal convex blocks.
///
/// `node_inputs`/`node_outputs`: value names per node (ORT binding order);
/// `supported[i]`: node `i` is executable by the interpreter. Returns groups
/// as lists of node indices (every supported node appears in exactly one
/// group; unsupported ones stay on the CPU EP).
pub fn convex_groups(
    num_nodes: usize,
    node_inputs: &[Vec<String>],
    node_outputs: &[Vec<String>],
    supported: &[bool],
) -> Vec<Vec<usize>> {
    use std::collections::HashMap;

    // --- producer→consumer edges on the full graph ---
    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (i, outs) in node_outputs.iter().enumerate() {
        for o in outs {
            if !o.is_empty() {
                producer.insert(o.as_str(), i);
            }
        }
    }
    // preds[v] / succs[u] (dedup not needed for correctness)
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for (v, ins) in node_inputs.iter().enumerate() {
        for name in ins {
            if let Some(&u) = producer.get(name.as_str()) {
                preds[v].push(u);
                succs[u].push(v);
                if supported[u] && supported[v] {
                    edges.push((u, v));
                }
            }
        }
    }

    // --- topological order (Kahn) ---
    let mut indeg: Vec<usize> = preds.iter().map(|p| p.len()).collect();
    let mut queue: Vec<usize> = (0..num_nodes).filter(|&v| indeg[v] == 0).collect();
    let mut topo = Vec::with_capacity(num_nodes);
    let mut head = 0;
    while head < queue.len() {
        let u = queue[head];
        head += 1;
        topo.push(u);
        for &v in &succs[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                queue.push(v);
            }
        }
    }
    // if the graph had a cycle (it should not) the residual nodes are appended
    // to the queue: anc/desc remain a safe approximation (a stricter test).
    if topo.len() < num_nodes {
        for v in 0..num_nodes {
            if !topo.contains(&v) {
                topo.push(v);
            }
        }
    }

    // --- ancestors / descendants (bitsets, excluding self) ---
    let w = words(num_nodes);
    let mut anc: Vec<Bits> = vec![vec![0u64; w]; num_nodes];
    for &v in &topo {
        for u in preds[v].clone() {
            let (av, au) = split_two(&mut anc, v, u);
            or_into(av, au);
            set_bit(av, u);
        }
    }
    let mut desc: Vec<Bits> = vec![vec![0u64; w]; num_nodes];
    for &u in topo.iter().rev() {
        for v in succs[u].clone() {
            let (du, dv) = split_two(&mut desc, u, v);
            or_into(du, dv);
            set_bit(du, v);
        }
    }

    // --- convex greedy union-find ---
    // state per cluster representative: members S, ancestor/descendant union.
    let mut uf = Uf::new(num_nodes);
    let mut s: Vec<Bits> = vec![vec![0u64; w]; num_nodes];
    let mut canc: Vec<Bits> = anc.clone();
    let mut cdesc: Vec<Bits> = desc.clone();
    for (i, &sup) in supported.iter().enumerate() {
        if sup {
            set_bit(&mut s[i], i);
        }
    }

    let mut ms = vec![0u64; w];
    let mut ma = vec![0u64; w];
    let mut md = vec![0u64; w];
    let mut scratch = Vec::new();
    let mut down: std::collections::HashSet<usize> = std::collections::HashSet::new();
    loop {
        let mut changed = false;
        for &(u, v) in &edges {
            let ra = uf.find(u);
            let rb = uf.find(v);
            if ra == rb {
                continue;
            }
            // resulting set (computed in reused buffers)
            for k in 0..w {
                ms[k] = s[ra][k] | s[rb][k];
                ma[k] = canc[ra][k] | canc[rb][k];
                md[k] = cdesc[ra][k] | cdesc[rb][k];
            }
            // The merge is legal only if the **quotient** graph stays acyclic:
            // no other cluster may lie both downstream and upstream of the
            // resulting set. The node-only test is insufficient — two convex
            // groups can close a cycle through distinct external nodes in the
            // two directions (see `keeps_the_quotient_graph_acyclic`).
            set_bits_outside(&md, &ms, &mut scratch);
            down.clear();
            for &x in &scratch {
                down.insert(uf.find(x));
            }
            set_bits_outside(&ma, &ms, &mut scratch);
            if scratch.iter().any(|&x| down.contains(&uf.find(x))) {
                continue;
            }
            // commit: merge rb into ra
            uf.parent[rb] = ra;
            s[ra].copy_from_slice(&ms);
            canc[ra].copy_from_slice(&ma);
            cdesc[ra].copy_from_slice(&md);
            changed = true;
        }
        if !changed {
            break;
        }
    }

    // --- collect groups (rep → members, by node index order) ---
    let mut by_rep: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, &sup) in supported.iter().enumerate() {
        if sup {
            by_rep.entry(uf.find(i)).or_default().push(i);
        }
    }
    by_rep.into_values().collect()
}

/// Disjoint mutable borrows of two distinct elements of a slice.
fn split_two<T>(v: &mut [T], a: usize, b: usize) -> (&mut T, &mut T) {
    assert!(a != b);
    if a < b {
        let (lo, hi) = v.split_at_mut(b);
        (&mut lo[a], &mut hi[0])
    } else {
        let (lo, hi) = v.split_at_mut(a);
        (&mut hi[0], &mut lo[b])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn merges_adjacent_supported_chain() {
        // 0 -> 1 -> 2, all supported: a single group.
        let ins = vec![names(&[]), names(&["a"]), names(&["b"])];
        let outs = vec![names(&["a"]), names(&["b"]), names(&["c"])];
        let sup = vec![true, true, true];
        let g = convex_groups(3, &ins, &outs, &sup);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0], vec![0, 1, 2]);
    }

    #[test]
    fn rejects_non_convex() {
        // 0(sup) -> 1(NOT sup) -> 2(sup) and 0 -> 2 directly.
        // Fusing 0 and 2 would trap 1 → must stay separate.
        let ins = vec![names(&[]), names(&["a"]), names(&["a", "b"])];
        let outs = vec![names(&["a"]), names(&["b"]), names(&["c"])];
        let sup = vec![true, false, true];
        let g = convex_groups(3, &ins, &outs, &sup);
        assert_eq!(g.len(), 2, "0 and 2 not fusable (1 in between)");
    }

    /// The quotient graph (contracted groups + unsupported nodes) must stay
    /// acyclic: ORT rejects the model otherwise.
    fn quotient_is_acyclic(
        num_nodes: usize,
        ins: &[Vec<String>],
        outs: &[Vec<String>],
        groups: &[Vec<usize>],
    ) -> bool {
        use std::collections::HashMap;
        let mut cluster: Vec<usize> = (0..num_nodes).collect();
        for (g, members) in groups.iter().enumerate() {
            for &m in members {
                cluster[m] = num_nodes + g;
            }
        }
        let mut producer: HashMap<&str, usize> = HashMap::new();
        for (i, o) in outs.iter().enumerate() {
            for name in o {
                producer.insert(name.as_str(), i);
            }
        }
        let total = num_nodes + groups.len();
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); total];
        let mut indeg = vec![0usize; total];
        for (v, i) in ins.iter().enumerate() {
            for name in i {
                if let Some(&u) = producer.get(name.as_str()) {
                    let (cu, cv) = (cluster[u], cluster[v]);
                    if cu != cv {
                        succ[cu].push(cv);
                        indeg[cv] += 1;
                    }
                }
            }
        }
        // Kahn: if nodes with in-degree > 0 remain, there is a cycle
        let mut queue: Vec<usize> = (0..total).filter(|&v| indeg[v] == 0).collect();
        let mut seen = 0;
        let mut head = 0;
        while head < queue.len() {
            let u = queue[head];
            head += 1;
            seen += 1;
            for &v in &succ[u].clone() {
                indeg[v] -= 1;
                if indeg[v] == 0 {
                    queue.push(v);
                }
            }
        }
        seen == total
    }

    #[test]
    fn keeps_the_quotient_graph_acyclic() {
        // Two groups, each convex on its own, linked in both directions through
        // distinct UNSUPPORTED nodes:
        //   A = {a1,a2,a3}   a1→a3, a2→a3
        //   B = {b1,b2,b3}   b1→b3, b2→b3
        //   a1 → x → b1      (x unsupported)
        //   b2 → y → a2      (y unsupported)
        // No external node lies between two members of the same group, so the
        // convexity-only test accepts both — but the quotient contains
        // A→x→B→y→A. Fusing both groups is illegal.
        let ins = vec![
            names(&[]),           // 0 a1
            names(&["y"]),        // 1 a2
            names(&["a1", "a2"]), // 2 a3
            names(&["x"]),        // 3 b1
            names(&[]),           // 4 b2
            names(&["b1", "b2"]), // 5 b3
            names(&["a1"]),       // 6 x  (unsupported)
            names(&["b2"]),       // 7 y  (unsupported)
        ];
        let outs = vec![
            names(&["a1"]),
            names(&["a2"]),
            names(&["a3"]),
            names(&["b1"]),
            names(&["b2"]),
            names(&["b3"]),
            names(&["x"]),
            names(&["y"]),
        ];
        let sup = vec![true, true, true, true, true, true, false, false];
        let groups = convex_groups(8, &ins, &outs, &sup);
        assert!(
            quotient_is_acyclic(8, &ins, &outs, &groups),
            "gruppi con quoziente ciclico: {groups:?}"
        );
    }

    #[test]
    fn merges_around_unsupported_side_branch() {
        // 0(sup) -> 1(sup) -> 2(sup); 1 -> 3(NOT sup) side branch.
        // 3 is not between two members → 0,1,2 are fusible.
        let ins = vec![names(&[]), names(&["a"]), names(&["b"]), names(&["b"])];
        let outs = vec![names(&["a"]), names(&["b"]), names(&["c"]), names(&["d"])];
        let sup = vec![true, true, true, false];
        let g = convex_groups(4, &ins, &outs, &sup);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0], vec![0, 1, 2]);
    }
}
