//! Tesseract-style text line detection
//!
//! This module implements Tesseract's text line detection algorithms:
//! - TextlineProjection: Projection profile-based line detection
//! - Row making: Grouping blobs into text rows
//! - Baseline fitting: Fitting baselines to rows

use crate::core::geometry::TBox;
use crate::recognition::tesseract_blob::BlobNBox;
use crate::utils::Result;

/// Text line projection (equivalent to Tesseract's TextlineProjection)
///
/// Builds a projection map representing local textline density by smearing
/// connected components horizontally and counting overlaps.
pub struct TextlineProjection {
    /// Scale factor (to achieve ~100 ppi resolution)
    scale_factor: u32,
    /// Projection image (scaled down)
    projection: Vec<Vec<u32>>,
    /// X origin in image coordinates
    x_origin: i32,
    /// Y origin in image coordinates
    y_origin: i32,
    /// Image width
    image_width: u32,
    /// Image height
    image_height: u32,
}

impl TextlineProjection {
    /// Create a new textline projection
    pub fn new(resolution: u32) -> Self {
        // Scale factor to achieve ~100 ppi
        let scale_factor = (resolution / 100).max(1);

        Self {
            scale_factor,
            projection: Vec::new(),
            x_origin: 0,
            y_origin: 0,
            image_width: 0,
            image_height: 0,
        }
    }

    /// Build projection profile from blobs
    pub fn construct_projection(
        &mut self,
        blobs: &[BlobNBox],
        image_width: u32,
        image_height: u32,
    ) -> Result<()> {
        self.image_width = image_width;
        self.image_height = image_height;
        self.x_origin = 0;
        self.y_origin = image_height as i32;

        // Calculate projection dimensions
        let proj_width = (image_width + self.scale_factor - 1) / self.scale_factor;
        let proj_height = (image_height + self.scale_factor - 1) / self.scale_factor;

        // Initialize projection
        self.projection = vec![vec![0; proj_width as usize]; proj_height as usize];

        // Project blobs onto the projection map
        for blob in blobs {
            self.project_blob(blob);
        }

        // Smooth the projection (block convolution)
        self.smooth_projection();

        Ok(())
    }

    /// Project a single blob onto the projection map
    fn project_blob(&mut self, blob: &BlobNBox) {
        let bbox = blob.bounding_box();
        let width = bbox.width() as u32;

        // Project blob horizontally (smear by width)
        let top = self.image_y_to_projection_y(bbox.top() as u32);
        let bottom = self.image_y_to_projection_y(bbox.bottom() as u32);

        for y in bottom..=top {
            if y < self.projection.len() as u32 {
                let left = self.image_x_to_projection_x(bbox.left() as u32);
                let right = self.image_x_to_projection_x(bbox.right() as u32);

                for x in left..=right {
                    if x < self.projection[y as usize].len() as u32 {
                        // Weight by blob width (like Tesseract)
                        self.projection[y as usize][x as usize] += width;
                    }
                }
            }
        }
    }

    /// Smooth projection using block convolution
    fn smooth_projection(&mut self) {
        if self.projection.is_empty() {
            return;
        }

        let height = self.projection.len();
        let width = if height > 0 {
            self.projection[0].len()
        } else {
            0
        };

        // Simple 3x3 box filter (blockconv)
        let mut smoothed = vec![vec![0u32; width]; height];

        for y in 0..height {
            for x in 0..width {
                let mut sum = 0u32;
                let mut count = 0u32;

                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let ny = (y as i32 + dy).max(0).min(height as i32 - 1) as usize;
                        let nx = (x as i32 + dx).max(0).min(width as i32 - 1) as usize;
                        sum += self.projection[ny][nx];
                        count += 1;
                    }
                }

                smoothed[y][x] = if count > 0 { sum / count } else { 0 };
            }
        }

        self.projection = smoothed;
    }

    /// Convert image X coordinate to projection X coordinate
    fn image_x_to_projection_x(&self, x: u32) -> u32 {
        (x / self.scale_factor).min(self.projection[0].len() as u32 - 1)
    }

    /// Convert image Y coordinate to projection Y coordinate
    fn image_y_to_projection_y(&self, y: u32) -> u32 {
        let proj_height = self.projection.len() as u32;
        // Y is inverted (0 at top in image, but we store bottom-up)
        let inverted_y = self.image_height - 1 - y;
        (inverted_y / self.scale_factor).min(proj_height - 1)
    }

    /// Evaluate if a box sits well on a textline
    pub fn evaluate_box(&self, bbox: &TBox) -> i32 {
        // Compute gradients at box edges
        let top_gradient = self.best_mean_gradient_in_row(
            bbox.left() as u32,
            bbox.right() as u32,
            bbox.top() as u32,
            true,
        );
        let bottom_gradient = -self.best_mean_gradient_in_row(
            bbox.left() as u32,
            bbox.right() as u32,
            bbox.bottom() as u32,
            false,
        );
        let left_gradient = self.best_mean_gradient_in_column(
            bbox.left() as u32,
            bbox.bottom() as u32,
            bbox.top() as u32,
            true,
        );
        let right_gradient = -self.best_mean_gradient_in_column(
            bbox.right() as u32,
            bbox.bottom() as u32,
            bbox.top() as u32,
            false,
        );

        // Clamp gradients to non-negative
        let top_clipped = top_gradient.max(0);
        let bottom_clipped = bottom_gradient.max(0);
        let left_clipped = left_gradient.max(0);
        let right_clipped = right_gradient.max(0);

        // Result is max vertical gradient minus max horizontal gradient
        // Positive means blob sits well on a horizontal textline
        std::cmp::max(top_clipped, bottom_clipped) - std::cmp::max(left_clipped, right_clipped)
    }

    /// Find best mean gradient in a horizontal row
    fn best_mean_gradient_in_row(&self, x1: u32, x2: u32, y: u32, increasing: bool) -> i32 {
        let proj_y = self.image_y_to_projection_y(y);
        if proj_y >= self.projection.len() as u32 {
            return 0;
        }

        let proj_x1 = self.image_x_to_projection_x(x1);
        let proj_x2 = self.image_x_to_projection_x(x2);

        if proj_x1 >= self.projection[proj_y as usize].len() as u32
            || proj_x2 >= self.projection[proj_y as usize].len() as u32
        {
            return 0;
        }

        // Sample gradient at multiple offsets
        let mut best_gradient = 0i32;

        for offset in -2..=2 {
            let sample_y = if proj_y as i32 + offset >= 0
                && (proj_y as i32 + offset) < self.projection.len() as i32
            {
                (proj_y as i32 + offset) as usize
            } else {
                continue;
            };

            let mut gradient_sum = 0i32;
            let mut count = 0;

            for x in proj_x1..=proj_x2 {
                if x < self.projection[sample_y].len() as u32 {
                    let value = self.projection[sample_y][x as usize] as i32;
                    if increasing {
                        gradient_sum += value;
                    } else {
                        gradient_sum -= value;
                    }
                    count += 1;
                }
            }

            if count > 0 {
                let gradient = gradient_sum / count as i32;
                if gradient.abs() > best_gradient.abs() {
                    best_gradient = gradient;
                }
            }
        }

        best_gradient
    }

    /// Find best mean gradient in a vertical column
    fn best_mean_gradient_in_column(&self, x: u32, y1: u32, y2: u32, increasing: bool) -> i32 {
        let proj_x = self.image_x_to_projection_x(x);
        if proj_x >= self.projection[0].len() as u32 {
            return 0;
        }

        let proj_y1 = self.image_y_to_projection_y(y1);
        let proj_y2 = self.image_y_to_projection_y(y2);

        let mut gradient_sum = 0i32;
        let mut count = 0;

        let start_y = proj_y1.min(proj_y2);
        let end_y = proj_y1.max(proj_y2);

        for y in start_y..=end_y {
            if y < self.projection.len() as u32 && proj_x < self.projection[y as usize].len() as u32
            {
                let value = self.projection[y as usize][proj_x as usize] as i32;
                if increasing {
                    gradient_sum += value;
                } else {
                    gradient_sum -= value;
                }
                count += 1;
            }
        }

        if count > 0 {
            gradient_sum / count as i32
        } else {
            0
        }
    }

    /// Find peaks in projection (text line centers)
    pub fn find_line_centers(&self, threshold_ratio: f32) -> Vec<u32> {
        if self.projection.is_empty() {
            return Vec::new();
        }

        // Find max value in projection
        let max_value = self
            .projection
            .iter()
            .flat_map(|row| row.iter())
            .max()
            .copied()
            .unwrap_or(0) as f32;

        let threshold = (max_value * threshold_ratio) as u32;

        let mut peaks = Vec::new();
        let height = self.projection.len();

        // Find peaks in each column (vertical projection)
        for x in 0..self.projection[0].len() {
            let mut in_peak = false;
            let mut peak_max = 0u32;
            let mut peak_max_y = 0usize;

            for y in 0..height {
                let value = self.projection[y][x];

                if value >= threshold {
                    if !in_peak {
                        in_peak = true;
                        peak_max = value;
                        peak_max_y = y;
                    } else if value > peak_max {
                        peak_max = value;
                        peak_max_y = y;
                    }
                } else if in_peak {
                    // End of peak
                    let image_y = self.projection_y_to_image_y(peak_max_y as u32);
                    peaks.push(image_y);
                    in_peak = false;
                }
            }

            // Handle peak at end
            if in_peak {
                let image_y = self.projection_y_to_image_y(peak_max_y as u32);
                peaks.push(image_y);
            }
        }

        // Remove duplicates and sort
        peaks.sort();
        peaks.dedup();

        peaks
    }

    /// Convert projection Y coordinate back to image Y coordinate
    fn projection_y_to_image_y(&self, proj_y: u32) -> u32 {
        let inverted_y = proj_y * self.scale_factor;
        self.image_height - 1 - inverted_y
    }
}

/// Text row (equivalent to Tesseract's TO_ROW)
#[derive(Debug, Clone)]
pub struct TextRow {
    /// Blobs in this row
    blobs: Vec<BlobNBox>,
    /// Baseline (y = mx + c)
    baseline_m: f32,
    baseline_c: f32,
    /// Mean blob height
    mean_height: f32,
    /// Line spacing
    line_spacing: f32,
    /// X-height (height of lowercase letters)
    x_height: f32,
}

impl TextRow {
    /// Create a new text row
    pub fn new() -> Self {
        Self {
            blobs: Vec::new(),
            baseline_m: 0.0,
            baseline_c: 0.0,
            mean_height: 0.0,
            line_spacing: 0.0,
            x_height: 0.0,
        }
    }

    /// Add a blob to this row
    pub fn add_blob(&mut self, blob: BlobNBox) {
        self.blobs.push(blob);
    }

    /// Fit baseline using least median squares (LMS)
    pub fn fit_baseline(&mut self) -> Result<()> {
        if self.blobs.is_empty() {
            return Ok(());
        }

        // Collect blob bottom centers
        let mut points = Vec::new();
        for blob in &self.blobs {
            let bbox = blob.bounding_box();
            let x = (bbox.left() + bbox.right()) / 2;
            let y = bbox.bottom();
            points.push((x as f32, y as f32));
        }

        // Simple linear regression for baseline
        let n = points.len() as f32;
        let sum_x: f32 = points.iter().map(|(x, _)| x).sum();
        let sum_y: f32 = points.iter().map(|(_, y)| y).sum();
        let sum_xy: f32 = points.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f32 = points.iter().map(|(x, _)| x * x).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-6 {
            self.baseline_m = 0.0;
            self.baseline_c = sum_y / n;
        } else {
            self.baseline_m = (n * sum_xy - sum_x * sum_y) / denominator;
            self.baseline_c = (sum_y * sum_x2 - sum_x * sum_xy) / denominator;
        }

        // Calculate mean height
        let heights: Vec<f32> = self
            .blobs
            .iter()
            .map(|b| b.bounding_box().height() as f32)
            .collect();
        self.mean_height = heights.iter().sum::<f32>() / heights.len() as f32;

        // Estimate x-height (simplified)
        self.x_height = self.mean_height * 0.7;
        self.line_spacing = self.mean_height * 1.5;

        Ok(())
    }

    /// Get blobs
    pub fn blobs(&self) -> &[BlobNBox] {
        &self.blobs
    }

    /// Get baseline slope
    pub fn baseline_slope(&self) -> f32 {
        self.baseline_m
    }

    /// Get baseline intercept
    pub fn baseline_intercept(&self) -> f32 {
        self.baseline_c
    }

    /// Get mean height
    pub fn mean_height(&self) -> f32 {
        self.mean_height
    }
}

impl Default for TextRow {
    fn default() -> Self {
        Self::new()
    }
}

/// Group blobs into text rows using projection profile method
pub fn group_blobs_into_rows(
    blobs: &[BlobNBox],
    image_width: u32,
    image_height: u32,
) -> Result<Vec<TextRow>> {
    if blobs.is_empty() {
        return Ok(Vec::new());
    }

    // Build projection profile
    let mut projection = TextlineProjection::new(300); // Assume 300 DPI
    projection.construct_projection(blobs, image_width, image_height)?;

    // Find line centers
    let line_centers = projection.find_line_centers(0.3);

    if line_centers.is_empty() {
        // Fallback to simple grouping
        return group_blobs_into_rows_simple(blobs);
    }

    // Group blobs by nearest line center
    let mut rows: Vec<TextRow> = vec![TextRow::new(); line_centers.len()];

    for blob in blobs {
        let bbox = blob.bounding_box();
        let blob_center_y = (bbox.top() + bbox.bottom()) / 2;

        // Find nearest line center
        let mut best_row_idx = 0;
        let mut min_distance = i32::MAX;

        for (idx, &line_y) in line_centers.iter().enumerate() {
            let distance = (blob_center_y - line_y as i32).abs();
            if distance < min_distance {
                min_distance = distance;
                best_row_idx = idx;
            }
        }

        // Check if blob is close enough to the line
        let line_y = line_centers[best_row_idx];
        let blob_height = bbox.height();
        let tolerance = blob_height.max(5) / 2;

        if (blob_center_y - line_y as i32).abs() <= tolerance as i32 {
            rows[best_row_idx].add_blob(blob.clone());
        }
    }

    // Remove empty rows and fit baselines
    let mut result_rows = Vec::new();
    for mut row in rows {
        if !row.blobs().is_empty() {
            // Sort blobs horizontally
            row.blobs
                .sort_by(|a, b| a.bounding_box().left().cmp(&b.bounding_box().left()));
            row.fit_baseline()?;
            result_rows.push(row);
        }
    }

    Ok(result_rows)
}

/// Simple row grouping fallback
fn group_blobs_into_rows_simple(blobs: &[BlobNBox]) -> Result<Vec<TextRow>> {
    if blobs.is_empty() {
        return Ok(Vec::new());
    }

    // Sort blobs by y-position
    let mut sorted_blobs = blobs.to_vec();
    sorted_blobs.sort_by(|a, b| {
        let a_center_y = (a.bounding_box().top() + a.bounding_box().bottom()) / 2;
        let b_center_y = (b.bounding_box().top() + b.bounding_box().bottom()) / 2;
        a_center_y.cmp(&b_center_y)
    });

    // Group blobs into rows
    let mut rows = Vec::new();
    let mut current_row = TextRow::new();
    let mut last_y = None;

    for blob in sorted_blobs {
        let center_y = (blob.bounding_box().top() + blob.bounding_box().bottom()) / 2;
        let height = blob.bounding_box().height();
        let tolerance = height.max(5) / 2;

        if let Some(last_y_pos) = last_y {
            if center_y > last_y_pos + tolerance {
                // New row
                if !current_row.blobs().is_empty() {
                    current_row.fit_baseline()?;
                    rows.push(current_row);
                    current_row = TextRow::new();
                }
            }
        }

        current_row.add_blob(blob);
        last_y = Some(center_y);
    }

    if !current_row.blobs().is_empty() {
        current_row.fit_baseline()?;
        rows.push(current_row);
    }

    Ok(rows)
}
