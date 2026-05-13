//! Union-Find (Disjoint Set) based Connected Component Labeling
//!
//! This module implements efficient connected component labeling using
//! the Union-Find data structure. It's much faster than recursive flood-fill
//! for large images and is the method used by Tesseract.
//!
//! References:
//! - "Efficient Connected Component Labeling" by Kesheng Wu, Ekow Otoo, Kenji Suzuki
//! - Tesseract's CCL implementation in ccnont.cpp

use image::GrayImage;

/// Union-Find (Disjoint Set) data structure for CCL
#[derive(Debug, Clone)]
pub struct UnionFind {
    /// Parent array where parent[i] is the parent of element i
    parent: Vec<usize>,
    /// Rank array for union by rank optimization
    rank: Vec<usize>,
    /// Size of each set (number of elements)
    size: Vec<usize>,
}

impl UnionFind {
    /// Create a new Union-Find structure with `n` elements
    pub fn new(n: usize) -> Self {
        let parent = (0..n).collect();
        let rank = vec![0; n];
        let size = vec![1; n];
        Self { parent, rank, size }
    }

    /// Create a new Union-Find structure with capacity for `n` elements
    pub fn with_capacity(n: usize) -> Self {
        let parent = Vec::with_capacity(n);
        let rank = Vec::with_capacity(n);
        let size = Vec::with_capacity(n);
        Self { parent, rank, size }
    }

    /// Find the representative (root) of the set containing `x`
    /// Uses path compression to optimize future queries
    #[inline]
    pub fn find(&mut self, x: usize) -> usize {
        if x >= self.parent.len() {
            return x;
        }
        if self.parent[x] != x {
            // Path compression: make every node on the path point directly to root
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Find without path compression (for internal use)
    #[inline]
    fn find_no_compression(&self, x: usize) -> usize {
        let mut current = x;
        while self.parent[current] != current {
            current = self.parent[current];
        }
        current
    }

    /// Merge the sets containing elements `x` and `y`
    /// Uses union by rank to keep the tree shallow
    #[inline]
    pub fn union(&mut self, x: usize, y: usize) {
        let x_root = self.find_no_compression(x);
        let y_root = self.find_no_compression(y);

        if x_root == y_root {
            return;
        }

        // Union by rank: attach smaller tree to larger tree
        if self.rank[x_root] < self.rank[y_root] {
            self.parent[x_root] = y_root;
            self.size[y_root] += self.size[x_root];
        } else if self.rank[x_root] > self.rank[y_root] {
            self.parent[y_root] = x_root;
            self.size[x_root] += self.size[y_root];
        } else {
            // Equal rank, choose one as new root
            self.parent[y_root] = x_root;
            self.rank[x_root] += 1;
            self.size[x_root] += self.size[y_root];
        }
    }

    /// Add a new element as a singleton set
    pub fn add_element(&mut self) -> usize {
        let idx = self.parent.len();
        self.parent.push(idx);
        self.rank.push(0);
        self.size.push(1);
        idx
    }

    /// Get the size of the set containing element `x`
    pub fn set_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        if root < self.size.len() {
            self.size[root]
        } else {
            1
        }
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Compress all paths in the Union-Find structure
    /// This should be called after all unions are done
    pub fn compress_all(&mut self) {
        for i in 0..self.parent.len() {
            let _ = self.find(i);
        }
    }
}

/// Connected component labeling result
#[derive(Debug, Clone)]
pub struct CclResult {
    /// Label for each pixel (0 = background, >0 = component ID)
    pub labels: Vec<Vec<u32>>,
    /// Number of connected components found
    pub num_components: usize,
    /// Bounding boxes for each component (1-indexed)
    pub bounding_boxes: Vec<ComponentBoundingBox>,
}

/// Bounding box for a connected component
#[derive(Debug, Clone, Copy)]
pub struct ComponentBoundingBox {
    /// Left coordinate (inclusive)
    pub left: u32,
    /// Top coordinate (inclusive)
    pub top: u32,
    /// Right coordinate (exclusive)
    pub right: u32,
    /// Bottom coordinate (exclusive)
    pub bottom: u32,
}

impl Default for ComponentBoundingBox {
    /// Returns an "empty" sentinel bbox: left/top = u32::MAX, right/bottom = 0.
    /// After calling `include` at least once the values become valid.
    fn default() -> Self {
        Self {
            left: u32::MAX,
            top: u32::MAX,
            right: 0,
            bottom: 0,
        }
    }
}

impl ComponentBoundingBox {
    /// Create a new bounding box
    pub fn new(left: u32, top: u32, right: u32, bottom: u32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    /// Returns true if no pixel has been included yet (sentinel state).
    pub fn is_empty(&self) -> bool {
        self.left == u32::MAX
    }

    /// Get width of the bounding box
    pub fn width(&self) -> u32 {
        if self.is_empty() {
            0
        } else {
            self.right.saturating_sub(self.left)
        }
    }

    /// Get height of the bounding box
    pub fn height(&self) -> u32 {
        if self.is_empty() {
            0
        } else {
            self.bottom.saturating_sub(self.top)
        }
    }

    /// Get area of the bounding box
    pub fn area(&self) -> u64 {
        self.width() as u64 * self.height() as u64
    }

    /// Expand the bounding box to include a point
    pub fn include(&mut self, x: u32, y: u32) {
        self.left = self.left.min(x);
        self.top = self.top.min(y);
        self.right = self.right.max(x + 1);
        self.bottom = self.bottom.max(y + 1);
    }
}

/// Perform connected component labeling on a binary image
///
/// Uses Union-Find based two-pass algorithm:
/// 1. First pass: assign provisional labels and record equivalences
/// 2. Second pass: resolve equivalences and assign final labels
///
/// # Arguments
/// * `image` - Binary image (0 = background, <128 = foreground)
///
/// # Returns
/// * `CclResult` containing labels, component count, and bounding boxes
pub fn connected_components_4connectivity(image: &GrayImage) -> CclResult {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return CclResult {
            labels: Vec::new(),
            num_components: 0,
            bounding_boxes: Vec::new(),
        };
    }

    let w = width as usize;
    let h = height as usize;

    // Initialize labels array (0 = background/unlabeled)
    let mut labels = vec![vec![0u32; w]; h];
    let mut uf = UnionFind::with_capacity((w * h) / 10); // Estimate components

    // Temporary storage for bounding boxes during first pass
    let mut temp_boxes: Vec<Option<ComponentBoundingBox>> = Vec::new();

    // First pass: scan and assign provisional labels
    for y in 0..h {
        for x in 0..w {
            // Check if current pixel is foreground (dark)
            if image.get_pixel(x as u32, y as u32)[0] >= 128 {
                continue; // Background pixel
            }

            // Check neighbors (top and left for 4-connectivity)
            let top_label = if y > 0 { labels[y - 1][x] } else { 0 };
            let left_label = if x > 0 { labels[y][x - 1] } else { 0 };

            match (top_label > 0, left_label > 0) {
                (false, false) => {
                    // No labeled neighbors - create new label
                    let new_label = uf.add_element() as u32 + 1;
                    labels[y][x] = new_label;
                    temp_boxes.push(Some(ComponentBoundingBox::new(
                        x as u32,
                        y as u32,
                        x as u32 + 1,
                        y as u32 + 1,
                    )));
                }
                (true, false) => {
                    // Only top neighbor has label
                    labels[y][x] = top_label;
                    let idx = (top_label - 1) as usize;
                    if idx < temp_boxes.len() {
                        if let Some(ref mut bbox) = temp_boxes[idx] {
                            bbox.include(x as u32, y as u32);
                        }
                    }
                }
                (false, true) => {
                    // Only left neighbor has label
                    labels[y][x] = left_label;
                    let idx = (left_label - 1) as usize;
                    if idx < temp_boxes.len() {
                        if let Some(ref mut bbox) = temp_boxes[idx] {
                            bbox.include(x as u32, y as u32);
                        }
                    }
                }
                (true, true) => {
                    // Both neighbors have labels - need to merge
                    labels[y][x] = top_label.min(left_label);

                    // Union the two labels
                    let top_idx = (top_label - 1) as usize;
                    let left_idx = (left_label - 1) as usize;

                    // Ensure Union-Find is large enough
                    while uf.len() <= top_idx.max(left_idx) {
                        uf.add_element();
                    }

                    uf.union(top_idx, left_idx);

                    // Update bounding boxes
                    if top_idx < temp_boxes.len() && left_idx < temp_boxes.len() {
                        let bbox_idx = labels[y][x] as usize - 1;
                        if bbox_idx < temp_boxes.len() {
                            if let Some(ref mut bbox) = temp_boxes[bbox_idx] {
                                bbox.include(x as u32, y as u32);
                            }
                        }
                    }
                }
            }
        }
    }

    // Compress all paths in Union-Find
    uf.compress_all();

    // Build final label mapping
    let mut label_mapping: Vec<u32> = vec![0; uf.len() + 1];
    let mut current_final_label = 1u32;

    for i in 0..uf.len() {
        let root = uf.find(i);
        if label_mapping[root] == 0 {
            label_mapping[root] = current_final_label;
            current_final_label += 1;
        }
        label_mapping[i] = label_mapping[root];
    }

    // Second pass: resolve equivalences and collect final bounding boxes
    let num_components = (current_final_label - 1) as usize;
    let mut final_boxes = vec![ComponentBoundingBox::default(); num_components + 1];

    for y in 0..h {
        for x in 0..w {
            let old_label = labels[y][x];
            if old_label == 0 {
                continue;
            }

            let old_idx = (old_label - 1) as usize;
            let new_label = if old_idx < label_mapping.len() {
                label_mapping[old_idx]
            } else {
                old_label
            };

            labels[y][x] = new_label;

            if new_label > 0 {
                let idx = new_label as usize;
                if idx <= num_components {
                    final_boxes[idx].include(x as u32, y as u32);
                }
            }
        }
    }

    CclResult {
        labels,
        num_components,
        bounding_boxes: final_boxes,
    }
}

/// Perform connected component labeling using 8-connectivity
///
/// 8-connectivity considers all 8 neighbors (including diagonals).
/// This is typically better for text characters.
pub fn connected_components_8connectivity(image: &GrayImage) -> CclResult {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return CclResult {
            labels: Vec::new(),
            num_components: 0,
            bounding_boxes: Vec::new(),
        };
    }

    let w = width as usize;
    let h = height as usize;

    // Initialize labels array
    let mut labels = vec![vec![0u32; w]; h];
    let mut uf = UnionFind::with_capacity((w * h) / 10);

    // Temporary storage for bounding boxes
    let mut temp_boxes: Vec<Option<ComponentBoundingBox>> = Vec::new();

    // Neighbor offsets for 8-connectivity (scan order: top-left, top, top-right, left)
    let neighbors = [(0isize, -1isize), (-1, -1), (-1, 0), (-1, 1)];

    // First pass
    for y in 0..h {
        for x in 0..w {
            // Check if current pixel is foreground
            if image.get_pixel(x as u32, y as u32)[0] >= 128 {
                continue;
            }

            // Collect labels of neighboring foreground pixels
            let mut neighbor_labels: Vec<u32> = Vec::new();

            for (dx, dy) in &neighbors {
                let nx = x as isize + dx;
                let ny = y as isize + dy;

                if nx >= 0 && ny >= 0 && nx < w as isize && ny < h as isize {
                    let neighbor_label = labels[ny as usize][nx as usize];
                    if neighbor_label > 0 {
                        neighbor_labels.push(neighbor_label);
                    }
                }
            }

            if neighbor_labels.is_empty() {
                // No labeled neighbors - create new label
                let new_label = uf.add_element() as u32 + 1;
                labels[y][x] = new_label;
                temp_boxes.push(Some(ComponentBoundingBox::new(
                    x as u32,
                    y as u32,
                    x as u32 + 1,
                    y as u32 + 1,
                )));
            } else {
                // Use minimum label from neighbors
                let min_label = *neighbor_labels.iter().min().unwrap();
                labels[y][x] = min_label;

                // Union all neighbor labels
                for &neighbor_label in &neighbor_labels {
                    let min_idx = (min_label - 1) as usize;
                    let neighbor_idx = (neighbor_label - 1) as usize;

                    // Ensure Union-Find is large enough
                    while uf.len() <= min_idx.max(neighbor_idx) {
                        uf.add_element();
                    }

                    uf.union(min_idx, neighbor_idx);
                }

                // Update bounding box
                let bbox_idx = (min_label - 1) as usize;
                if bbox_idx < temp_boxes.len() {
                    if let Some(ref mut bbox) = temp_boxes[bbox_idx] {
                        bbox.include(x as u32, y as u32);
                    }
                }
            }
        }
    }

    // Compress all paths
    uf.compress_all();

    // Build final label mapping
    let mut label_mapping: Vec<u32> = vec![0; uf.len() + 1];
    let mut current_final_label = 1u32;

    for i in 0..uf.len() {
        let root = uf.find(i);
        if label_mapping[root] == 0 {
            label_mapping[root] = current_final_label;
            current_final_label += 1;
        }
        label_mapping[i] = label_mapping[root];
    }

    // Second pass
    let num_components = (current_final_label - 1) as usize;
    let mut final_boxes = vec![ComponentBoundingBox::default(); num_components + 1];

    for y in 0..h {
        for x in 0..w {
            let old_label = labels[y][x];
            if old_label == 0 {
                continue;
            }

            let old_idx = (old_label - 1) as usize;
            let new_label = if old_idx < label_mapping.len() {
                label_mapping[old_idx]
            } else {
                old_label
            };

            labels[y][x] = new_label;

            if new_label > 0 {
                let idx = new_label as usize;
                if idx <= num_components {
                    final_boxes[idx].include(x as u32, y as u32);
                }
            }
        }
    }

    CclResult {
        labels,
        num_components,
        bounding_boxes: final_boxes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(10);

        // Initially each element is its own parent
        for i in 0..10 {
            assert_eq!(uf.find(i), i);
        }

        // Union some elements
        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(1, 2); // Now 0, 1, 2, 3 are in same set

        assert_eq!(uf.find(0), uf.find(1));
        assert_eq!(uf.find(1), uf.find(2));
        assert_eq!(uf.find(2), uf.find(3));
        assert_ne!(uf.find(0), uf.find(4));
    }

    #[test]
    fn test_union_find_path_compression() {
        let mut uf = UnionFind::new(100);

        // Create a chain: 0-1-2-3-...-99
        for i in 0..99 {
            uf.union(i, i + 1);
        }

        // After path compression, all finds should be fast
        let root = uf.find(0);
        assert_eq!(uf.find(99), root);
        assert_eq!(uf.find(50), root);
    }

    #[test]
    fn test_ccl_single_pixel() {
        // Create a white image (all pixels = 255)
        let mut img = GrayImage::from_pixel(10, 10, Luma([255u8]));

        // Verify the pixel is white initially
        assert_eq!(img.get_pixel(0, 0)[0], 255);

        // Set one pixel to black
        img.put_pixel(5, 5, Luma([0u8]));

        // Verify the pixel is now black
        assert_eq!(img.get_pixel(5, 5)[0], 0);

        let result = connected_components_4connectivity(&img);
        assert_eq!(result.num_components, 1);
        assert_eq!(result.labels[5][5], 1);
        assert_eq!(result.bounding_boxes[1].left, 5);
        assert_eq!(result.bounding_boxes[1].top, 5);
        assert_eq!(result.bounding_boxes[1].right, 6);
        assert_eq!(result.bounding_boxes[1].bottom, 6);
    }

    #[test]
    fn test_ccl_two_components() {
        // Create a white image
        let mut img = GrayImage::from_pixel(20, 20, Luma([255u8]));

        // First component: 2x2 square at (2,2)
        for y in 2..4 {
            for x in 2..4 {
                img.put_pixel(x, y, Luma([0u8]));
            }
        }

        // Second component: 3x3 square at (10,10)
        for y in 10..13 {
            for x in 10..13 {
                img.put_pixel(x, y, Luma([0u8]));
            }
        }

        let result = connected_components_4connectivity(&img);
        assert_eq!(result.num_components, 2);
    }

    #[test]
    fn test_ccl_diagonal_4connectivity() {
        // Create a white image
        let mut img = GrayImage::from_pixel(10, 10, Luma([255u8]));

        // Two pixels touching diagonally
        img.put_pixel(5, 5, Luma([0u8]));
        img.put_pixel(6, 6, Luma([0u8]));

        let result = connected_components_4connectivity(&img);
        // With 4-connectivity, diagonal pixels are separate components
        assert_eq!(result.num_components, 2);
    }

    #[test]
    fn test_ccl_diagonal_8connectivity() {
        // Create a white image
        let mut img = GrayImage::from_pixel(10, 10, Luma([255u8]));

        // Two pixels touching diagonally
        img.put_pixel(5, 5, Luma([0u8]));
        img.put_pixel(6, 6, Luma([0u8]));

        let result = connected_components_8connectivity(&img);
        // With 8-connectivity, diagonal pixels are same component
        assert_eq!(result.num_components, 1);
    }

    #[test]
    fn test_bounding_box_expansion() {
        let mut bbox = ComponentBoundingBox::new(10, 10, 11, 11);

        bbox.include(15, 15);
        bbox.include(5, 5);

        assert_eq!(bbox.left, 5);
        assert_eq!(bbox.top, 5);
        assert_eq!(bbox.right, 16);
        assert_eq!(bbox.bottom, 16);
        assert_eq!(bbox.width(), 11);
        assert_eq!(bbox.height(), 11);
    }
}
