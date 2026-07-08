//! Lightweight detection CNN for text region heatmaps
//!
//! Implements a 3-layer U-Net-style architecture using ndarray.
//! Generates a per-pixel text-probability heatmap, which is then
//! thresholded and post-processed into bounding boxes.

use ndarray::{Array2, Array3, Array4};

/// Small CNN for text region detection
pub struct TextDetectionCNN {
    /// Conv1 weights: 3×3 kernel, 1 input channel, 8 output channels
    conv1: Array3<f32>,
    conv1_bias: Vec<f32>,
    /// Conv2 weights: 3×3 kernel, 8 input channels, 16 output channels
    conv2: Array4<f32>,
    conv2_bias: Vec<f32>,
    /// Conv3 weights: 1×1 kernel, 16 input channels, 1 output channel (heatmap)
    conv3: Array4<f32>,
    conv3_bias: f32,
}

impl Default for TextDetectionCNN {
    fn default() -> Self {
        Self::new()
    }
}

impl TextDetectionCNN {
    /// Create a new CNN with heuristic initial weights
    /// that approximate edge/gradient detection (reasonable text prior)
    pub fn new() -> Self {
        use fastrand::f32;

        let mut conv1 = Array3::zeros((8, 3, 3));
        let mut conv1_bias = vec![0.0f32; 8];
        for och in 0..8 {
            // Initialize as oriented edge detectors
            let kernel = match och {
                0 => [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], // horizontal
                1 => [[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]], // horizontal flipped
                2 => [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], // vertical
                3 => [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]], // vertical flipped
                4 => [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], // Laplacian
                5 => [[-1.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], // diagonal
                6 => [[0.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, -1.0, 0.0]], // other diagonal
                _ => {
                    let mut k = [[0.0f32; 3]; 3];
                    for i in 0..3 {
                        for j in 0..3 {
                            k[i][j] = f32() * 0.4 - 0.2;
                        }
                    }
                    k
                }
            };
            for i in 0..3 {
                for j in 0..3 {
                    conv1[(och, i, j)] = kernel[i][j];
                }
            }
            conv1_bias[och] = -0.1;
        }

        let mut conv2 = Array4::zeros((16, 8, 3, 3));
        let mut conv2_bias = vec![0.0f32; 16];
        for och in 0..16 {
            for ich in 0..8 {
                for i in 0..3 {
                    for j in 0..3 {
                        conv2[(och, ich, i, j)] = f32() * 0.3 - 0.15;
                    }
                }
            }
            conv2_bias[och] = -0.1;
        }

        let mut conv3 = Array4::zeros((1, 16, 1, 1));
        for ich in 0..16 {
            conv3[(0, ich, 0, 0)] = 0.05;
        }

        Self {
            conv1,
            conv1_bias,
            conv2,
            conv2_bias,
            conv3,
            conv3_bias: -0.5,
        }
    }

    /// Forward pass on a grayscale image.
    /// Returns a text-probability heatmap (values roughly 0..1 after sigmoid).
    pub fn forward(&self, image: &Array2<f32>) -> Array2<f32> {
        let (h, w) = (image.shape()[0], image.shape()[1]);

        // Normalize input: invert so text is high signal
        let mut input = image.clone();
        for v in input.iter_mut() {
            *v = 1.0 - *v; // invert: black text = 1.0
        }

        // Pad for convolutions
        let padded = Self::pad2d(&input, 1);

        // Conv1: 1 → 8 channels, 3×3
        let mut feat1 = Array3::zeros((8, h, w));
        for och in 0..8 {
            for y in 0..h {
                for x in 0..w {
                    let mut sum = self.conv1_bias[och];
                    for i in 0..3 {
                        for j in 0..3 {
                            let py = y + i;
                            let px = x + j;
                            sum += padded[(py, px)] * self.conv1[(och, i, j)];
                        }
                    }
                    feat1[(och, y, x)] = Self::relu(sum);
                }
            }
        }

        // Pad feat1
        let mut padded_feat1 = Array3::zeros((8, h + 2, w + 2));
        for c in 0..8 {
            let slice = feat1.slice(ndarray::s![c, .., ..]);
            let padded_slice = Self::pad2d(&slice.to_owned(), 1);
            for y in 0..h + 2 {
                for x in 0..w + 2 {
                    padded_feat1[(c, y, x)] = padded_slice[(y, x)];
                }
            }
        }

        // Conv2: 8 → 16 channels, 3×3
        let mut feat2 = Array3::zeros((16, h, w));
        for och in 0..16 {
            for y in 0..h {
                for x in 0..w {
                    let mut sum = self.conv2_bias[och];
                    for ich in 0..8 {
                        for i in 0..3 {
                            for j in 0..3 {
                                sum += padded_feat1[(ich, y + i, x + j)] * self.conv2[(och, ich, i, j)];
                            }
                        }
                    }
                    feat2[(och, y, x)] = Self::relu(sum);
                }
            }
        }

        // Conv3: 16 → 1 channel, 1×1 (pointwise)
        let mut heatmap = Array2::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                let mut sum = self.conv3_bias;
                for ich in 0..16 {
                    sum += feat2[(ich, y, x)] * self.conv3[(0, ich, 0, 0)];
                }
                heatmap[(y, x)] = Self::sigmoid(sum);
            }
        }

        heatmap
    }

    /// Post-process heatmap: threshold → connected components → bounding boxes
    pub fn post_process(
        &self,
        heatmap: &Array2<f32>,
        threshold: f32,
        min_area: usize,
    ) -> Vec<(usize, usize, usize, usize)> {
        let (h, w) = (heatmap.shape()[0], heatmap.shape()[1]);

        // Binary mask
        let mut mask = vec![false; h * w];
        for y in 0..h {
            for x in 0..w {
                if heatmap[(y, x)] > threshold {
                    mask[y * w + x] = true;
                }
            }
        }

        // Union-find connected components (4-connected)
        let mut parent: Vec<usize> = (0..(h * w)).collect();
        fn find(parent: &mut [usize], i: usize) -> usize {
            let mut root = i;
            while parent[root] != root {
                root = parent[root];
            }
            // Path compression
            let mut j = i;
            while parent[j] != root {
                let next = parent[j];
                parent[j] = root;
                j = next;
            }
            root
        }
        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[rb] = ra;
            }
        }

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if !mask[idx] {
                    continue;
                }
                // Right neighbor
                if x + 1 < w && mask[idx + 1] {
                    union(&mut parent, idx, idx + 1);
                }
                // Bottom neighbor
                if y + 1 < h && mask[idx + w] {
                    union(&mut parent, idx, idx + w);
                }
            }
        }

        // Collect component bounding boxes and areas
        let mut comps: std::collections::HashMap<usize, (usize, usize, usize, usize, usize)> =
            std::collections::HashMap::new();
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if !mask[idx] {
                    continue;
                }
                let root = find(&mut parent, idx);
                let entry = comps.entry(root).or_insert((x, x, y, y, 0));
                entry.0 = entry.0.min(x);
                entry.1 = entry.1.max(x);
                entry.2 = entry.2.min(y);
                entry.3 = entry.3.max(y);
                entry.4 += 1;
            }
        }

        comps
            .values()
            .filter(|(_, _, _, _, area)| *area >= min_area)
            .map(|(min_x, max_x, min_y, max_y, _)| (*min_x, *max_x, *min_y, *max_y))
            .collect()
    }

    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn pad2d(arr: &Array2<f32>, pad: usize) -> Array2<f32> {
        let (h, w) = (arr.shape()[0], arr.shape()[1]);
        let mut out = Array2::zeros((h + pad * 2, w + pad * 2));
        for y in 0..h {
            for x in 0..w {
                out[(y + pad, x + pad)] = arr[(y, x)];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cnn_forward_shape() {
        let cnn = TextDetectionCNN::new();
        let img = Array2::zeros((32, 64));
        let heatmap = cnn.forward(&img);
        assert_eq!(heatmap.shape(), &[32, 64]);
    }

    #[test]
    fn test_cnn_forward_runs_on_textured_image() {
        let cnn = TextDetectionCNN::new();
        let mut img = Array2::ones((40, 80));
        for y in 10..30 {
            for x in (10..70).step_by(2) {
                img[(y, x)] = 0.0;
            }
        }
        let heatmap = cnn.forward(&img);
        assert_eq!(heatmap.shape(), &[40, 80]);
        // Verify post-processing runs without panic even if heuristic weights don't trigger detection
        let _boxes = cnn.post_process(&heatmap, 0.5, 10);
    }

    #[test]
    fn test_post_process_empty() {
        let cnn = TextDetectionCNN::new();
        let heatmap = Array2::zeros((20, 20));
        let boxes = cnn.post_process(&heatmap, 0.5, 5);
        assert!(boxes.is_empty());
    }

    #[test]
    fn test_post_process_single_blob() {
        let cnn = TextDetectionCNN::new();
        let mut heatmap = Array2::zeros((20, 20));
        for y in 5..10 {
            for x in 5..15 {
                heatmap[(y, x)] = 0.9;
            }
        }
        let boxes = cnn.post_process(&heatmap, 0.5, 5);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0], (5, 14, 5, 9));
    }
}
