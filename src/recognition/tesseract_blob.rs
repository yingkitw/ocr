//! Tesseract-style blob detection and analysis
//!
//! This module implements Tesseract's blob detection algorithms:
//! - C_OUTLINE: Chain-coded outlines
//! - TBLOB: Text blobs with outlines
//! - BLOBNBOX: Blob bounding boxes with analysis

use crate::core::geometry::{ICoord, TBox};
use crate::utils::Result;
use image::GrayImage;

/// Direction codes for chain coding (4-directional)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainDir {
    Left = 0,  // (-1, 0)
    Up = 1,    // (0, -1)
    Right = 2, // (1, 0)
    Down = 3,  // (0, 1)
}

impl ChainDir {
    /// Convert to coordinate offset
    pub fn to_offset(self) -> (i32, i32) {
        match self {
            ChainDir::Left => (-1, 0),
            ChainDir::Up => (0, -1),
            ChainDir::Right => (1, 0),
            ChainDir::Down => (0, 1),
        }
    }

    /// Get direction from offset
    pub fn from_offset(dx: i32, dy: i32) -> Option<Self> {
        match (dx, dy) {
            (-1, 0) => Some(ChainDir::Left),
            (0, -1) => Some(ChainDir::Up),
            (1, 0) => Some(ChainDir::Right),
            (0, 1) => Some(ChainDir::Down),
            _ => None,
        }
    }
}

/// Chain-coded outline (equivalent to Tesseract's C_OUTLINE)
#[derive(Debug, Clone)]
pub struct ChainOutline {
    /// Starting position
    pub start: ICoord,
    /// Chain code steps (4-directional)
    pub steps: Vec<ChainDir>,
    /// Bounding box
    pub bounding_box: TBox,
    /// Child outlines (holes)
    pub children: Vec<ChainOutline>,
    /// Is this an inverse outline (white on black)?
    pub is_inverse: bool,
}

impl ChainOutline {
    /// Create a new chain outline
    pub fn new(start: ICoord, steps: Vec<ChainDir>, is_inverse: bool) -> Self {
        let mut bbox = TBox::new(start.x, start.y, start.x, start.y);
        let mut pos = start;

        for &step in &steps {
            let (dx, dy) = step.to_offset();
            pos = ICoord::new(pos.x + dx, pos.y + dy);
            bbox = bbox.union(&TBox::new(pos.x, pos.y, pos.x, pos.y));
        }

        Self {
            start,
            steps,
            bounding_box: bbox,
            children: Vec::new(),
            is_inverse,
        }
    }

    /// Get bounding box
    pub fn bounding_box(&self) -> &TBox {
        &self.bounding_box
    }

    /// Get path length
    pub fn path_length(&self) -> usize {
        self.steps.len()
    }

    /// Get perimeter
    pub fn perimeter(&self) -> f32 {
        self.steps.len() as f32
    }

    /// Get area (using shoelace formula)
    pub fn area(&self) -> f32 {
        if self.steps.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        let mut pos = self.start;

        for &step in &self.steps {
            let next_pos = {
                let (dx, dy) = step.to_offset();
                ICoord::new(pos.x + dx, pos.y + dy)
            };

            // Shoelace formula
            area += (pos.x as f32) * (next_pos.y as f32);
            area -= (next_pos.x as f32) * (pos.y as f32);

            pos = next_pos;
        }

        area.abs() / 2.0
    }

    /// Add child outline (hole)
    pub fn add_child(&mut self, child: ChainOutline) {
        self.children.push(child);
    }
}

/// Text blob with outlines (equivalent to Tesseract's TBLOB)
#[derive(Debug, Clone)]
pub struct TBlob {
    /// Outer outlines
    outlines: Vec<ChainOutline>,
    /// Bounding box
    bounding_box: TBox,
}

impl TBlob {
    /// Create a new blob from outlines
    pub fn new(outlines: Vec<ChainOutline>) -> Self {
        let mut bbox = if let Some(first) = outlines.first() {
            first.bounding_box().clone()
        } else {
            TBox::new(0, 0, 0, 0)
        };

        for outline in &outlines {
            bbox = bbox.union(outline.bounding_box());
        }

        Self {
            outlines,
            bounding_box: bbox,
        }
    }

    /// Get bounding box
    pub fn bounding_box(&self) -> &TBox {
        &self.bounding_box
    }

    /// Get outlines
    pub fn outlines(&self) -> &[ChainOutline] {
        &self.outlines
    }

    /// Compute blob area
    pub fn area(&self) -> f32 {
        let mut total_area = 0.0;

        for outline in &self.outlines {
            total_area += outline.area();
            // Subtract holes
            for child in &outline.children {
                total_area -= child.area();
            }
        }

        total_area
    }

    /// Compute blob perimeter
    pub fn perimeter(&self) -> f32 {
        self.outlines.iter().map(|o| o.perimeter()).sum()
    }
}

/// Blob bounding box with analysis (equivalent to Tesseract's BLOBNBOX)
#[derive(Debug, Clone)]
pub struct BlobNBox {
    /// The blob
    blob: TBlob,
    /// Bounding box
    bounding_box: TBox,
    /// Area
    area: f32,
    /// Perimeter
    perimeter: f32,
    /// Stroke width (estimated)
    stroke_width: f32,
    /// Aspect ratio
    aspect_ratio: f32,
    /// Is this a diacritic?
    is_diacritic: bool,
    /// Is this joined to another blob?
    joined: bool,
}

impl BlobNBox {
    /// Create a new BlobNBox from a TBlob
    pub fn new(blob: TBlob) -> Self {
        let bbox = blob.bounding_box().clone();
        let area = blob.area();
        let perimeter = blob.perimeter();
        let width = bbox.width() as f32;
        let height = bbox.height() as f32;
        let aspect_ratio = if height > 0.0 { width / height } else { 1.0 };

        // Estimate stroke width (simplified)
        let stroke_width = if perimeter > 0.0 {
            (area * 2.0) / perimeter
        } else {
            1.0
        };

        Self {
            blob,
            bounding_box: bbox,
            area,
            perimeter,
            stroke_width,
            aspect_ratio,
            is_diacritic: false,
            joined: false,
        }
    }

    /// Get bounding box
    pub fn bounding_box(&self) -> &TBox {
        &self.bounding_box
    }

    /// Get blob
    pub fn blob(&self) -> &TBlob {
        &self.blob
    }

    /// Get area
    pub fn area(&self) -> f32 {
        self.area
    }

    /// Get aspect ratio
    pub fn aspect_ratio(&self) -> f32 {
        self.aspect_ratio
    }

    /// Get stroke width
    pub fn stroke_width(&self) -> f32 {
        self.stroke_width
    }

    /// Check if this is a diacritic
    pub fn is_diacritic(&self) -> bool {
        self.is_diacritic
    }

    /// Set diacritic flag
    pub fn set_diacritic(&mut self, value: bool) {
        self.is_diacritic = value;
    }

    /// Check if joined
    pub fn is_joined(&self) -> bool {
        self.joined
    }

    /// Set joined flag
    pub fn set_joined(&mut self, value: bool) {
        self.joined = value;
    }
}

/// Extract chain-coded outlines from binary image using edge following
pub fn extract_outlines(image: &GrayImage) -> Result<Vec<ChainOutline>> {
    let (width, height) = image.dimensions();
    let mut visited = vec![vec![false; height as usize]; width as usize];
    let mut outlines = Vec::new();

    // Find all outlines by following edges
    for y in 0..height {
        for x in 0..width {
            if visited[x as usize][y as usize] {
                continue;
            }

            let pixel = image.get_pixel(x, y);
            // Look for black pixels (text)
            if pixel[0] < 128 {
                if let Some(outline) = follow_edge(image, x, y, &mut visited)? {
                    outlines.push(outline);
                }
            }
        }
    }

    Ok(outlines)
}

/// Follow an edge to create a chain-coded outline
fn follow_edge(
    image: &GrayImage,
    start_x: u32,
    start_y: u32,
    visited: &mut Vec<Vec<bool>>,
) -> Result<Option<ChainOutline>> {
    let (width, height) = image.dimensions();
    let mut steps = Vec::new();
    let mut pos = ICoord::new(start_x as i32, start_y as i32);
    let start = pos;
    let mut first = true;

    // Use Moore neighborhood tracing (8-connected)
    // Simplified to 4-connected for now
    let directions = [
        ChainDir::Right,
        ChainDir::Down,
        ChainDir::Left,
        ChainDir::Up,
    ];

    loop {
        if pos.x < 0 || pos.x >= width as i32 || pos.y < 0 || pos.y >= height as i32 {
            break;
        }

        let px = pos.x as usize;
        let py = pos.y as usize;

        if visited[px][py] && !first {
            break;
        }

        visited[px][py] = true;

        if first {
            first = false;
        }

        // Find next edge pixel
        let mut found = false;
        for &dir in &directions {
            let (dx, dy) = dir.to_offset();
            let next_x = pos.x + dx;
            let next_y = pos.y + dy;

            if next_x < 0 || next_x >= width as i32 || next_y < 0 || next_y >= height as i32 {
                continue;
            }

            let next_pixel = image.get_pixel(next_x as u32, next_y as u32);

            // Check if this is an edge pixel (transition from black to white or vice versa)
            let current_pixel = image.get_pixel(pos.x as u32, pos.y as u32);
            let is_edge = (current_pixel[0] < 128) != (next_pixel[0] < 128);

            if is_edge
                && (!visited[next_x as usize][next_y as usize]
                    || (next_x == start.x && next_y == start.y))
            {
                steps.push(dir);
                pos = ICoord::new(next_x, next_y);
                found = true;
                break;
            }
        }

        if !found {
            break;
        }

        // Check if we've completed the loop
        if pos.x == start.x && pos.y == start.y && steps.len() > 2 {
            break;
        }

        // Prevent infinite loops
        if steps.len() > (width * height) as usize {
            break;
        }
    }

    if steps.len() < 4 {
        return Ok(None);
    }

    let is_inverse = {
        let pixel = image.get_pixel(start.x as u32, start.y as u32);
        pixel[0] >= 128
    };

    Ok(Some(ChainOutline::new(start, steps, is_inverse)))
}

/// Convert outlines to blobs
pub fn outlines_to_blobs(outlines: Vec<ChainOutline>) -> Vec<TBlob> {
    // Group outlines into blobs (outer outlines with their holes)
    let mut blobs = Vec::new();
    let mut used = vec![false; outlines.len()];

    for (i, outline) in outlines.iter().enumerate() {
        if used[i] || outline.is_inverse {
            continue;
        }

        used[i] = true;
        let mut blob_outlines = vec![outline.clone()];

        // Find holes (inverse outlines) inside this outline
        for (j, other) in outlines.iter().enumerate() {
            if used[j] || !other.is_inverse {
                continue;
            }

            // Check if other is inside outline
            if is_outline_inside(&outline.bounding_box, &other.bounding_box) {
                blob_outlines[0].add_child(other.clone());
                used[j] = true;
            }
        }

        blobs.push(TBlob::new(blob_outlines));
    }

    blobs
}

/// Check if one bounding box is inside another
fn is_outline_inside(outer: &TBox, inner: &TBox) -> bool {
    outer.left() <= inner.left()
        && outer.right() >= inner.right()
        && outer.top() >= inner.top()
        && outer.bottom() <= inner.bottom()
}
