//! Geometry types and operations for OCR processing
//!
//! This module provides coordinate systems, bounding boxes, and geometric operations
//! migrated from Tesseract's `points.h` and `rect.h`.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Image dimension type (width, height, coordinates)
///
/// This corresponds to Tesseract's `TDimension` type.
/// Uses `i32` for compatibility with modern systems and large images.
pub type TDimension = i32;

/// Floating point type used for calculations
///
/// This corresponds to Tesseract's `TFloat` type.
pub type TFloat = f32;

/// Integer coordinate point
///
/// This corresponds to Tesseract's `ICOORD` class.
/// Represents a point with integer coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ICoord {
    pub x: TDimension,
    pub y: TDimension,
}

impl ICoord {
    /// Create a new integer coordinate
    pub fn new(x: TDimension, y: TDimension) -> Self {
        Self { x, y }
    }

    /// Create a zero coordinate
    pub fn zero() -> Self {
        Self { x: 0, y: 0 }
    }

    /// Get the x coordinate
    pub fn x(&self) -> TDimension {
        self.x
    }

    /// Get the y coordinate
    pub fn y(&self) -> TDimension {
        self.y
    }

    /// Set the x coordinate
    pub fn set_x(&mut self, x: TDimension) {
        self.x = x;
    }

    /// Set the y coordinate
    pub fn set_y(&mut self, y: TDimension) {
        self.y = y;
    }

    /// Calculate the squared length of the vector
    pub fn sqlength(&self) -> f32 {
        (self.x * self.x + self.y * self.y) as f32
    }

    /// Calculate the length of the vector
    pub fn length(&self) -> f32 {
        self.sqlength().sqrt()
    }

    /// Calculate the squared distance to another point
    pub fn pt_to_pt_sqdist(&self, other: &ICoord) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy) as f32
    }

    /// Calculate the distance to another point
    pub fn pt_to_pt_dist(&self, other: &ICoord) -> f32 {
        self.pt_to_pt_sqdist(other).sqrt()
    }

    /// Calculate the angle of the vector
    pub fn angle(&self) -> f32 {
        (self.y as f32).atan2(self.x as f32)
    }

    /// Set coordinates with shrinking to fit if needed
    pub fn set_with_shrink(&mut self, x: i32, y: i32) {
        self.x = x.clamp(TDimension::MIN, TDimension::MAX);
        self.y = y.clamp(TDimension::MIN, TDimension::MAX);
    }

    /// Rotate the coordinate by a float vector
    pub fn rotate(&mut self, vec: &FCoord) {
        let new_x = (self.x as f32 * vec.x() - self.y as f32 * vec.y()).round() as TDimension;
        let new_y = (self.y as f32 * vec.x() + self.x as f32 * vec.y()).round() as TDimension;
        self.x = new_x;
        self.y = new_y;
    }
}

impl fmt::Display for ICoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

// Arithmetic operations for ICoord
impl Add for ICoord {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl AddAssign for ICoord {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl Sub for ICoord {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl SubAssign for ICoord {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl Mul<TDimension> for ICoord {
    type Output = Self;

    fn mul(self, scale: TDimension) -> Self {
        Self {
            x: self.x * scale,
            y: self.y * scale,
        }
    }
}

impl Mul<ICoord> for TDimension {
    type Output = ICoord;

    fn mul(self, coord: ICoord) -> ICoord {
        coord * self
    }
}

impl MulAssign<TDimension> for ICoord {
    fn mul_assign(&mut self, scale: TDimension) {
        self.x *= scale;
        self.y *= scale;
    }
}

impl Div<TDimension> for ICoord {
    type Output = Self;

    fn div(self, scale: TDimension) -> Self {
        Self {
            x: self.x / scale,
            y: self.y / scale,
        }
    }
}

impl DivAssign<TDimension> for ICoord {
    fn div_assign(&mut self, scale: TDimension) {
        self.x /= scale;
        self.y /= scale;
    }
}

impl Neg for ICoord {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

/// Float coordinate point
///
/// This corresponds to Tesseract's `FCOORD` class.
/// Represents a point with floating-point coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FCoord {
    pub x: TFloat,
    pub y: TFloat,
}

impl FCoord {
    /// Create a new float coordinate
    pub fn new(x: TFloat, y: TFloat) -> Self {
        Self { x, y }
    }

    /// Create a zero coordinate
    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Create from an integer coordinate
    pub fn from_icoord(icoord: &ICoord) -> Self {
        Self {
            x: icoord.x as TFloat,
            y: icoord.y as TFloat,
        }
    }

    /// Get the x coordinate
    pub fn x(&self) -> TFloat {
        self.x
    }

    /// Get the y coordinate
    pub fn y(&self) -> TFloat {
        self.y
    }

    /// Set the x coordinate
    pub fn set_x(&mut self, x: TFloat) {
        self.x = x;
    }

    /// Set the y coordinate
    pub fn set_y(&mut self, y: TFloat) {
        self.y = y;
    }

    /// Calculate the squared length of the vector
    pub fn sqlength(&self) -> TFloat {
        self.x * self.x + self.y * self.y
    }

    /// Calculate the length of the vector
    pub fn length(&self) -> TFloat {
        self.sqlength().sqrt()
    }

    /// Calculate the squared distance to another point
    pub fn pt_to_pt_sqdist(&self, other: &FCoord) -> TFloat {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Calculate the distance to another point
    pub fn pt_to_pt_dist(&self, other: &FCoord) -> TFloat {
        self.pt_to_pt_sqdist(other).sqrt()
    }

    /// Calculate the angle of the vector
    pub fn angle(&self) -> TFloat {
        self.y.atan2(self.x)
    }

    /// Normalize the vector to unit length
    pub fn normalize(&mut self) -> bool {
        let len = self.length();
        if len > 0.0 {
            self.x /= len;
            self.y /= len;
            true
        } else {
            false
        }
    }

    /// Rotate the coordinate by a float vector
    pub fn rotate(&mut self, vec: &FCoord) {
        let new_x = self.x * vec.x - self.y * vec.y;
        let new_y = self.y * vec.x + self.x * vec.y;
        self.x = new_x;
        self.y = new_y;
    }

    /// Unrotate the coordinate (undo a rotation)
    pub fn unrotate(&mut self, vec: &FCoord) {
        self.rotate(&FCoord::new(vec.x, -vec.y));
    }
}

impl fmt::Display for FCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2})", self.x, self.y)
    }
}

// Arithmetic operations for FCoord
impl Add for FCoord {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl AddAssign for FCoord {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl Sub for FCoord {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl SubAssign for FCoord {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl Mul<TFloat> for FCoord {
    type Output = Self;

    fn mul(self, scale: TFloat) -> Self {
        Self {
            x: self.x * scale,
            y: self.y * scale,
        }
    }
}

impl Mul<FCoord> for TFloat {
    type Output = FCoord;

    fn mul(self, coord: FCoord) -> FCoord {
        coord * self
    }
}

impl MulAssign<TFloat> for FCoord {
    fn mul_assign(&mut self, scale: TFloat) {
        self.x *= scale;
        self.y *= scale;
    }
}

impl Div<TFloat> for FCoord {
    type Output = Self;

    fn div(self, scale: TFloat) -> Self {
        if scale == 0.0 {
            panic!("Division by zero");
        }
        Self {
            x: self.x / scale,
            y: self.y / scale,
        }
    }
}

impl DivAssign<TFloat> for FCoord {
    fn div_assign(&mut self, scale: TFloat) {
        if scale == 0.0 {
            panic!("Division by zero");
        }
        self.x /= scale;
        self.y /= scale;
    }
}

impl Neg for FCoord {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

/// Bounding box for OCR elements
///
/// This corresponds to Tesseract's `TBOX` class.
/// Represents a rectangular bounding box with integer coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TBox {
    /// Bottom-left corner
    pub bot_left: ICoord,
    /// Top-right corner
    pub top_right: ICoord,
}

impl TBox {
    /// Create a null (empty) bounding box
    pub fn null() -> Self {
        Self {
            bot_left: ICoord::new(TDimension::MAX, TDimension::MAX),
            top_right: ICoord::new(TDimension::MIN, TDimension::MIN),
        }
    }

    /// Create a bounding box from two corner points
    pub fn from_corners(pt1: ICoord, pt2: ICoord) -> Self {
        let left = pt1.x.min(pt2.x);
        let right = pt1.x.max(pt2.x);
        let bottom = pt1.y.min(pt2.y);
        let top = pt1.y.max(pt2.y);

        Self {
            bot_left: ICoord::new(left, bottom),
            top_right: ICoord::new(right, top),
        }
    }

    /// Create a bounding box from coordinates
    pub fn new(left: TDimension, bottom: TDimension, right: TDimension, top: TDimension) -> Self {
        Self {
            bot_left: ICoord::new(left, bottom),
            top_right: ICoord::new(right, top),
        }
    }

    /// Create a bounding box around a float coordinate
    pub fn from_fcoord(pt: &FCoord) -> Self {
        Self {
            bot_left: ICoord::new(pt.x.floor() as TDimension, pt.y.floor() as TDimension),
            top_right: ICoord::new(pt.x.ceil() as TDimension, pt.y.ceil() as TDimension),
        }
    }

    /// Check if the box is null (empty)
    pub fn is_null(&self) -> bool {
        self.left() >= self.right() || self.top() <= self.bottom()
    }

    /// Get the top coordinate
    pub fn top(&self) -> TDimension {
        self.top_right.y
    }

    /// Set the top coordinate
    pub fn set_top(&mut self, y: TDimension) {
        self.top_right.set_y(y);
    }

    /// Get the bottom coordinate
    pub fn bottom(&self) -> TDimension {
        self.bot_left.y
    }

    /// Set the bottom coordinate
    pub fn set_bottom(&mut self, y: TDimension) {
        self.bot_left.set_y(y);
    }

    /// Get the left coordinate
    pub fn left(&self) -> TDimension {
        self.bot_left.x
    }

    /// Set the left coordinate
    pub fn set_left(&mut self, x: TDimension) {
        self.bot_left.set_x(x);
    }

    /// Get the right coordinate
    pub fn right(&self) -> TDimension {
        self.top_right.x
    }

    /// Set the right coordinate
    pub fn set_right(&mut self, x: TDimension) {
        self.top_right.set_x(x);
    }

    /// Get the x coordinate of the center
    pub fn x_middle(&self) -> TDimension {
        (self.bot_left.x + self.top_right.x) / 2
    }

    /// Get the y coordinate of the center
    pub fn y_middle(&self) -> TDimension {
        (self.bot_left.y + self.top_right.y) / 2
    }

    /// Get the bottom-left corner
    pub fn bot_left(&self) -> ICoord {
        self.bot_left
    }

    /// Get the bottom-right corner
    pub fn bot_right(&self) -> ICoord {
        ICoord::new(self.top_right.x, self.bot_left.y)
    }

    /// Get the top-left corner
    pub fn top_left(&self) -> ICoord {
        ICoord::new(self.bot_left.x, self.top_right.y)
    }

    /// Get the top-right corner
    pub fn top_right(&self) -> ICoord {
        self.top_right
    }

    /// Get the height of the box
    pub fn height(&self) -> TDimension {
        if !self.is_null() {
            self.top_right.y - self.bot_left.y
        } else {
            0
        }
    }

    /// Get the width of the box
    pub fn width(&self) -> TDimension {
        if !self.is_null() {
            self.top_right.x - self.bot_left.x
        } else {
            0
        }
    }

    /// Get the area of the box
    pub fn area(&self) -> i64 {
        if !self.is_null() {
            self.width() as i64 * self.height() as i64
        } else {
            0
        }
    }

    /// Pad the box by the given amounts
    pub fn pad(&mut self, xpad: TDimension, ypad: TDimension) {
        let pad = ICoord::new(xpad, ypad);
        self.bot_left = self.bot_left - pad;
        self.top_right = self.top_right + pad;
    }

    /// Move the box by a vector
    pub fn move_by(&mut self, vec: ICoord) {
        self.bot_left = self.bot_left + vec;
        self.top_right = self.top_right + vec;
    }

    /// Move the box by a float vector
    pub fn move_by_fcoord(&mut self, vec: &FCoord) {
        self.bot_left
            .set_x((self.bot_left.x as f32 + vec.x).floor() as TDimension);
        self.bot_left
            .set_y((self.bot_left.y as f32 + vec.y).floor() as TDimension);
        self.top_right
            .set_x((self.top_right.x as f32 + vec.x).ceil() as TDimension);
        self.top_right
            .set_y((self.top_right.y as f32 + vec.y).ceil() as TDimension);
    }

    /// Scale the box by a factor
    pub fn scale(&mut self, factor: f32) {
        self.bot_left
            .set_x((self.bot_left.x as f32 * factor).floor() as TDimension);
        self.bot_left
            .set_y((self.bot_left.y as f32 * factor).floor() as TDimension);
        self.top_right
            .set_x((self.top_right.x as f32 * factor).ceil() as TDimension);
        self.top_right
            .set_y((self.top_right.y as f32 * factor).ceil() as TDimension);
    }

    /// Scale the box by a vector
    pub fn scale_by_vector(&mut self, vec: &FCoord) {
        self.bot_left
            .set_x((self.bot_left.x as f32 * vec.x).floor() as TDimension);
        self.bot_left
            .set_y((self.bot_left.y as f32 * vec.y).floor() as TDimension);
        self.top_right
            .set_x((self.top_right.x as f32 * vec.x).ceil() as TDimension);
        self.top_right
            .set_y((self.top_right.y as f32 * vec.y).ceil() as TDimension);
    }

    /// Check if a point is contained in the box
    pub fn contains_point(&self, pt: &FCoord) -> bool {
        pt.x >= self.bot_left.x as f32
            && pt.x <= self.top_right.x as f32
            && pt.y >= self.bot_left.y as f32
            && pt.y <= self.top_right.y as f32
    }

    /// Check if another box is contained in this box
    pub fn contains_box(&self, other: &TBox) -> bool {
        self.contains_point(&FCoord::from_icoord(&other.bot_left))
            && self.contains_point(&FCoord::from_icoord(&other.top_right))
    }

    /// Check if two boxes overlap
    pub fn overlaps(&self, other: &TBox) -> bool {
        other.bot_left.x <= self.top_right.x
            && other.top_right.x >= self.bot_left.x
            && other.bot_left.y <= self.top_right.y
            && other.top_right.y >= self.bot_left.y
    }

    /// Check if two boxes have major overlap (more than half)
    pub fn major_overlap(&self, other: &TBox) -> bool {
        let x_overlap =
            other.top_right.x.min(self.top_right.x) - other.bot_left.x.max(self.bot_left.x);
        let y_overlap =
            other.top_right.y.min(self.top_right.y) - other.bot_left.y.max(self.bot_left.y);

        x_overlap * 2 >= other.width().min(self.width())
            && y_overlap * 2 >= other.height().min(self.height())
    }

    /// Check if boxes overlap on x-axis
    pub fn x_overlap(&self, other: &TBox) -> bool {
        other.bot_left.x <= self.top_right.x && other.top_right.x >= self.bot_left.x
    }

    /// Check if boxes overlap on y-axis
    pub fn y_overlap(&self, other: &TBox) -> bool {
        other.bot_left.y <= self.top_right.y && other.top_right.y >= self.bot_left.y
    }

    /// Get the horizontal gap between boxes (negative if overlapping)
    pub fn x_gap(&self, other: &TBox) -> TDimension {
        other.bot_left.x.max(self.bot_left.x) - other.top_right.x.min(self.top_right.x)
    }

    /// Get the vertical gap between boxes (negative if overlapping)
    pub fn y_gap(&self, other: &TBox) -> TDimension {
        other.bot_left.y.max(self.bot_left.y) - other.top_right.y.min(self.top_right.y)
    }

    /// Get the intersection of two boxes
    pub fn intersection(&self, other: &TBox) -> TBox {
        if !self.overlaps(other) {
            return TBox::null();
        }

        TBox::new(
            self.bot_left.x.max(other.bot_left.x),
            self.bot_left.y.max(other.bot_left.y),
            self.top_right.x.min(other.top_right.x),
            self.top_right.y.min(other.top_right.y),
        )
    }

    /// Get the bounding union of two boxes
    pub fn bounding_union(&self, other: &TBox) -> TBox {
        TBox::new(
            self.bot_left.x.min(other.bot_left.x),
            self.bot_left.y.min(other.bot_left.y),
            self.top_right.x.max(other.top_right.x),
            self.top_right.y.max(other.top_right.y),
        )
    }

    /// Set the box to given coordinates
    pub fn set_coords(
        &mut self,
        x_min: TDimension,
        y_min: TDimension,
        x_max: TDimension,
        y_max: TDimension,
    ) {
        self.bot_left.set_x(x_min);
        self.bot_left.set_y(y_min);
        self.top_right.set_x(x_max);
        self.top_right.set_y(y_max);
    }

    /// Union of two bounding boxes
    pub fn union(&self, other: &TBox) -> TBox {
        self.bounding_union(other)
    }

    /// Check if a point is contained in the box
    pub fn contains(&self, point: ICoord) -> bool {
        point.x >= self.bot_left.x
            && point.x <= self.top_right.x
            && point.y >= self.bot_left.y
            && point.y <= self.top_right.y
    }
}

impl fmt::Display for TBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({},{})->({},{})",
            self.left(),
            self.bottom(),
            self.right(),
            self.top()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icoord_creation() {
        let coord = ICoord::new(10, 20);
        assert_eq!(coord.x(), 10);
        assert_eq!(coord.y(), 20);
    }

    #[test]
    fn test_icoord_arithmetic() {
        let a = ICoord::new(10, 20);
        let b = ICoord::new(5, 15);

        assert_eq!(a + b, ICoord::new(15, 35));
        assert_eq!(a - b, ICoord::new(5, 5));
        assert_eq!(a * 2, ICoord::new(20, 40));
    }

    #[test]
    fn test_fcoord_creation() {
        let coord = FCoord::new(10.5, 20.7);
        assert_eq!(coord.x(), 10.5);
        assert_eq!(coord.y(), 20.7);
    }

    #[test]
    fn test_fcoord_arithmetic() {
        let a = FCoord::new(10.0, 20.0);
        let b = FCoord::new(5.0, 15.0);

        assert_eq!(a + b, FCoord::new(15.0, 35.0));
        assert_eq!(a - b, FCoord::new(5.0, 5.0));
        assert_eq!(a * 2.0, FCoord::new(20.0, 40.0));
    }

    #[test]
    fn test_tbox_creation() {
        let bbox = TBox::new(0, 0, 100, 200);
        assert_eq!(bbox.left(), 0);
        assert_eq!(bbox.bottom(), 0);
        assert_eq!(bbox.right(), 100);
        assert_eq!(bbox.top(), 200);
        assert_eq!(bbox.width(), 100);
        assert_eq!(bbox.height(), 200);
        assert_eq!(bbox.area(), 20000);
    }

    #[test]
    fn test_tbox_overlap() {
        let a = TBox::new(0, 0, 100, 100);
        let b = TBox::new(50, 50, 150, 150);
        let c = TBox::new(200, 200, 300, 300);

        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_tbox_intersection() {
        let a = TBox::new(0, 0, 100, 100);
        let b = TBox::new(50, 50, 150, 150);
        let intersection = a.intersection(&b);

        assert_eq!(intersection.left(), 50);
        assert_eq!(intersection.bottom(), 50);
        assert_eq!(intersection.right(), 100);
        assert_eq!(intersection.top(), 100);
    }
}
