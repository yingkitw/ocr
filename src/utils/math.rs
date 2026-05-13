//! Mathematical utilities for MiniOCR

use nalgebra::{DMatrix, Point2, Point3, Vector2, Vector3};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

/// 2D point type
pub type Point2D = Point2<f32>;

/// 3D point type
pub type Point3D = Point3<f32>;

/// 2D vector type
pub type Vector2D = Vector2<f32>;

/// Bounding box type (for mathematical operations)
///
/// Note: For OCR text elements, use `core::text::BoundingBox` instead
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MathBoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl MathBoundingBox {
    /// Create a new bounding box
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Get the center point of the bounding box
    pub fn center(&self) -> Point2D {
        Point2D::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Check if a point is inside the bounding box
    pub fn contains(&self, point: Point2D) -> bool {
        point.x >= self.x
            && point.x <= self.x + self.width
            && point.y >= self.y
            && point.y <= self.y + self.height
    }

    /// Check if this bounding box intersects with another
    pub fn intersects(&self, other: &MathBoundingBox) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }

    /// Get the area of the bounding box
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

/// Legacy alias for backward compatibility
#[deprecated(note = "Use MathBoundingBox or core::text::BoundingBox instead")]
pub type BoundingBox = MathBoundingBox;

/// 3D vector type
pub type Vector3D = Vector3<f32>;

/// 2D matrix type
pub type Matrix2D = DMatrix<f32>;

/// 2D array type
pub type Array2D = Array2<f32>;

/// 3D array type
pub type Array3D = Array3<f32>;

/// 2D array view type
pub type ArrayView2D<'a> = ArrayView2<'a, f32>;

/// 3D array view type
pub type ArrayView3D<'a> = ArrayView3<'a, f32>;

/// Convert degrees to radians
#[inline]
pub fn deg_to_rad(degrees: f32) -> f32 {
    degrees * std::f32::consts::PI / 180.0
}

/// Convert radians to degrees
#[inline]
pub fn rad_to_deg(radians: f32) -> f32 {
    radians * 180.0 / std::f32::consts::PI
}

/// Clamp a value between min and max
#[inline]
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

/// Linear interpolation between two values
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Calculate the distance between two 2D points
#[inline]
pub fn distance_2d(p1: Point2D, p2: Point2D) -> f32 {
    ((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt()
}

/// Calculate the angle between two 2D vectors
#[inline]
pub fn angle_2d(v1: Vector2D, v2: Vector2D) -> f32 {
    let dot = v1.dot(&v2);
    let mag1 = v1.magnitude();
    let mag2 = v2.magnitude();

    if mag1 == 0.0 || mag2 == 0.0 {
        0.0
    } else {
        (dot / (mag1 * mag2)).acos()
    }
}

/// Calculate the cross product of two 2D vectors
#[inline]
pub fn cross_2d(v1: Vector2D, v2: Vector2D) -> f32 {
    v1.x * v2.y - v1.y * v2.x
}

/// Check if a point is inside a rectangle
#[inline]
pub fn point_in_rect(point: Point2D, rect_min: Point2D, rect_max: Point2D) -> bool {
    point.x >= rect_min.x && point.x <= rect_max.x && point.y >= rect_min.y && point.y <= rect_max.y
}

/// Calculate the area of a rectangle
#[inline]
pub fn rect_area(min: Point2D, max: Point2D) -> f32 {
    (max.x - min.x) * (max.y - min.y)
}

/// Calculate the intersection area of two rectangles
pub fn rect_intersection_area(
    rect1_min: Point2D,
    rect1_max: Point2D,
    rect2_min: Point2D,
    rect2_max: Point2D,
) -> f32 {
    let left = rect1_min.x.max(rect2_min.x);
    let top = rect1_min.y.max(rect2_min.y);
    let right = rect1_max.x.min(rect2_max.x);
    let bottom = rect1_max.y.min(rect2_max.y);

    if left < right && top < bottom {
        (right - left) * (bottom - top)
    } else {
        0.0
    }
}

/// Calculate the IoU (Intersection over Union) of two rectangles
pub fn rect_iou(
    rect1_min: Point2D,
    rect1_max: Point2D,
    rect2_min: Point2D,
    rect2_max: Point2D,
) -> f32 {
    let intersection = rect_intersection_area(rect1_min, rect1_max, rect2_min, rect2_max);
    let union = rect_area(rect1_min, rect1_max) + rect_area(rect2_min, rect2_max) - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}
