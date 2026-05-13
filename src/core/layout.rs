//! Layout analysis data structures for MiniOCR

use crate::core::text::BoundingBox;
use crate::utils::Point2D;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Page layout analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutResult {
    /// Page dimensions
    pub page_size: PageSize,
    /// Detected blocks
    pub blocks: Vec<Block>,
    /// Detected text regions
    pub text_regions: Vec<TextRegion>,
    /// Detected images
    pub images: Vec<ImageRegion>,
    /// Detected tables
    pub tables: Vec<Table>,
    /// Reading order
    pub reading_order: ReadingOrder,
    /// Page orientation
    pub orientation: PageOrientation,
    /// Confidence score
    pub confidence: f32,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl LayoutResult {
    /// Create a new layout result
    pub fn new(page_size: PageSize) -> Self {
        Self {
            page_size,
            blocks: Vec::new(),
            text_regions: Vec::new(),
            images: Vec::new(),
            tables: Vec::new(),
            reading_order: ReadingOrder::TopToBottom,
            orientation: PageOrientation::Portrait,
            confidence: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Get all text blocks
    pub fn text_blocks(&self) -> Vec<&Block> {
        self.blocks
            .iter()
            .filter(|b| b.block_type == BlockType::Text)
            .collect()
    }

    /// Get all image blocks
    pub fn image_blocks(&self) -> Vec<&Block> {
        self.blocks
            .iter()
            .filter(|b| b.block_type == BlockType::Image)
            .collect()
    }

    /// Get all table blocks
    pub fn table_blocks(&self) -> Vec<&Block> {
        self.blocks
            .iter()
            .filter(|b| b.block_type == BlockType::Table)
            .collect()
    }
}

/// Page size information
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageSize {
    /// Page width in pixels
    pub width: u32,
    /// Page height in pixels
    pub height: u32,
    /// Page DPI
    pub dpi: u32,
}

impl PageSize {
    /// Create a new page size
    pub fn new(width: u32, height: u32, dpi: u32) -> Self {
        Self { width, height, dpi }
    }

    /// Get aspect ratio
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }

    /// Check if page is landscape
    pub fn is_landscape(&self) -> bool {
        self.width > self.height
    }

    /// Check if page is portrait
    pub fn is_portrait(&self) -> bool {
        self.height > self.width
    }
}

/// Block in the page layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    /// Block identifier
    pub id: String,
    /// Block type
    pub block_type: BlockType,
    /// Bounding box
    pub bounding_box: BoundingBox,
    /// Block content
    pub content: BlockContent,
    /// Block properties
    pub properties: BlockProperties,
    /// Confidence score
    pub confidence: f32,
}

impl Block {
    /// Create a new block
    pub fn new(id: String, block_type: BlockType, bounding_box: BoundingBox) -> Self {
        Self {
            id,
            block_type,
            bounding_box,
            content: BlockContent::Empty,
            properties: BlockProperties::default(),
            confidence: 0.0,
        }
    }

    /// Get block area
    pub fn area(&self) -> u32 {
        self.bounding_box.area() as u32
    }

    /// Get block center
    pub fn center(&self) -> Point2D {
        self.bounding_box.center()
    }
}

/// Block type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockType {
    /// Text block
    Text,
    /// Image block
    Image,
    /// Table block
    Table,
    /// Header block
    Header,
    /// Footer block
    Footer,
    /// Sidebar block
    Sidebar,
    /// Unknown block type
    Unknown,
}

/// Block content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockContent {
    /// Empty block
    Empty,
    /// Text content
    Text(String),
    /// Image content
    Image { path: String, format: String },
    /// Table content
    Table {
        rows: Vec<Vec<String>>,
        headers: Vec<String>,
    },
    /// Mixed content
    Mixed(Vec<BlockContent>),
}

/// Block properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockProperties {
    /// Block priority (for reading order)
    pub priority: u32,
    /// Block level (for hierarchy)
    pub level: u32,
    /// Block alignment
    pub alignment: TextAlignment,
    /// Block margins
    pub margins: Margins,
    /// Block padding
    pub padding: Margins,
    /// Block background color
    pub background_color: Option<Color>,
    /// Block text color
    pub text_color: Option<Color>,
}

impl Default for BlockProperties {
    fn default() -> Self {
        Self {
            priority: 0,
            level: 0,
            alignment: TextAlignment::Left,
            margins: Margins::zero(),
            padding: Margins::zero(),
            background_color: None,
            text_color: None,
        }
    }
}

/// Text region in the page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRegion {
    /// Region identifier
    pub id: String,
    /// Bounding box
    pub bounding_box: BoundingBox,
    /// Text content
    pub text: String,
    /// Text properties
    pub properties: TextRegionProperties,
    /// Confidence score
    pub confidence: f32,
}

impl TextRegion {
    /// Create a new text region
    pub fn new(id: String, bounding_box: BoundingBox, text: String) -> Self {
        Self {
            id,
            bounding_box,
            text,
            properties: TextRegionProperties::default(),
            confidence: 0.0,
        }
    }
}

/// Text region properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRegionProperties {
    /// Font family
    pub font_family: Option<String>,
    /// Font size
    pub font_size: f32,
    /// Font weight
    pub font_weight: FontWeight,
    /// Font style
    pub font_style: FontStyle,
    /// Text color
    pub text_color: Option<Color>,
    /// Background color
    pub background_color: Option<Color>,
    /// Line height
    pub line_height: f32,
    /// Letter spacing
    pub letter_spacing: f32,
    /// Word spacing
    pub word_spacing: f32,
}

impl Default for TextRegionProperties {
    fn default() -> Self {
        Self {
            font_family: None,
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            font_style: FontStyle::Normal,
            text_color: None,
            background_color: None,
            line_height: 1.0,
            letter_spacing: 0.0,
            word_spacing: 0.0,
        }
    }
}

/// Image region in the page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRegion {
    /// Region identifier
    pub id: String,
    /// Bounding box
    pub bounding_box: BoundingBox,
    /// Image properties
    pub properties: ImageRegionProperties,
    /// Confidence score
    pub confidence: f32,
}

impl ImageRegion {
    /// Create a new image region
    pub fn new(id: String, bounding_box: BoundingBox) -> Self {
        Self {
            id,
            bounding_box,
            properties: ImageRegionProperties::default(),
            confidence: 0.0,
        }
    }
}

/// Image region properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRegionProperties {
    /// Image format
    pub format: String,
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Image DPI
    pub dpi: u32,
    /// Image quality
    pub quality: f32,
    /// Image orientation
    pub orientation: ImageOrientation,
}

impl Default for ImageRegionProperties {
    fn default() -> Self {
        Self {
            format: "unknown".to_string(),
            width: 0,
            height: 0,
            dpi: 72,
            quality: 1.0,
            orientation: ImageOrientation::Normal,
        }
    }
}

/// Table in the page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    /// Table identifier
    pub id: String,
    /// Bounding box
    pub bounding_box: BoundingBox,
    /// Table structure
    pub structure: TableStructure,
    /// Table properties
    pub properties: TableProperties,
    /// Confidence score
    pub confidence: f32,
}

impl Table {
    /// Create a new table
    pub fn new(id: String, bounding_box: BoundingBox) -> Self {
        Self {
            id,
            bounding_box,
            structure: TableStructure::default(),
            properties: TableProperties::default(),
            confidence: 0.0,
        }
    }
}

/// Table structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStructure {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub columns: usize,
    /// Table cells
    pub cells: Vec<Vec<TableCell>>,
    /// Table headers
    pub headers: Vec<String>,
}

impl Default for TableStructure {
    fn default() -> Self {
        Self {
            rows: 0,
            columns: 0,
            cells: Vec::new(),
            headers: Vec::new(),
        }
    }
}

/// Table cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCell {
    /// Cell content
    pub content: String,
    /// Cell bounding box
    pub bounding_box: BoundingBox,
    /// Row span
    pub row_span: usize,
    /// Column span
    pub column_span: usize,
    /// Cell properties
    pub properties: CellProperties,
}

/// Cell properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellProperties {
    /// Cell alignment
    pub alignment: TextAlignment,
    /// Cell background color
    pub background_color: Option<Color>,
    /// Cell text color
    pub text_color: Option<Color>,
    /// Cell borders
    pub borders: Borders,
}

impl Default for CellProperties {
    fn default() -> Self {
        Self {
            alignment: TextAlignment::Left,
            background_color: None,
            text_color: None,
            borders: Borders::default(),
        }
    }
}

/// Table properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableProperties {
    /// Table alignment
    pub alignment: TextAlignment,
    /// Table borders
    pub borders: Borders,
    /// Table background color
    pub background_color: Option<Color>,
    /// Table header background color
    pub header_background_color: Option<Color>,
}

impl Default for TableProperties {
    fn default() -> Self {
        Self {
            alignment: TextAlignment::Left,
            borders: Borders::default(),
            background_color: None,
            header_background_color: None,
        }
    }
}

/// Reading order enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadingOrder {
    /// Top to bottom, left to right
    TopToBottom,
    /// Left to right, top to bottom
    LeftToRight,
    /// Right to left, top to bottom
    RightToLeft,
    /// Bottom to top, left to right
    BottomToTop,
    /// Multi-column layout
    MultiColumn,
}

/// Page orientation enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PageOrientation {
    /// Portrait orientation
    Portrait,
    /// Landscape orientation
    Landscape,
    /// Upside down portrait
    PortraitUpsideDown,
    /// Upside down landscape
    LandscapeUpsideDown,
}

/// Text alignment enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextAlignment {
    /// Left aligned
    Left,
    /// Right aligned
    Right,
    /// Center aligned
    Center,
    /// Justified
    Justified,
}

/// Font weight enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FontWeight {
    /// Normal weight
    Normal,
    /// Bold weight
    Bold,
    /// Light weight
    Light,
    /// Medium weight
    Medium,
    /// Heavy weight
    Heavy,
}

/// Font style enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FontStyle {
    /// Normal style
    Normal,
    /// Italic style
    Italic,
    /// Oblique style
    Oblique,
}

/// Image orientation enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageOrientation {
    /// Normal orientation
    Normal,
    /// Rotated 90 degrees clockwise
    Rotated90,
    /// Rotated 180 degrees
    Rotated180,
    /// Rotated 270 degrees clockwise
    Rotated270,
}

/// Color representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Color {
    /// Red component (0-255)
    pub r: u8,
    /// Green component (0-255)
    pub g: u8,
    /// Blue component (0-255)
    pub b: u8,
    /// Alpha component (0-255)
    pub a: u8,
}

impl Color {
    /// Create a new color
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create a color from RGB values
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Create a color from RGBA values
    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

/// Margins structure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Margins {
    /// Top margin
    pub top: u32,
    /// Right margin
    pub right: u32,
    /// Bottom margin
    pub bottom: u32,
    /// Left margin
    pub left: u32,
}

impl Margins {
    /// Create new margins
    pub fn new(top: u32, right: u32, bottom: u32, left: u32) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
        }
    }

    /// Create zero margins
    pub fn zero() -> Self {
        Self {
            top: 0,
            right: 0,
            bottom: 0,
            left: 0,
        }
    }

    /// Create uniform margins
    pub fn uniform(value: u32) -> Self {
        Self {
            top: value,
            right: value,
            bottom: value,
            left: value,
        }
    }
}

/// Borders structure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Borders {
    /// Top border
    pub top: Border,
    /// Right border
    pub right: Border,
    /// Bottom border
    pub bottom: Border,
    /// Left border
    pub left: Border,
}

impl Default for Borders {
    fn default() -> Self {
        Self {
            top: Border::default(),
            right: Border::default(),
            bottom: Border::default(),
            left: Border::default(),
        }
    }
}

/// Border structure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Border {
    /// Border width
    pub width: u32,
    /// Border color
    pub color: Color,
    /// Border style
    pub style: BorderStyle,
}

impl Default for Border {
    fn default() -> Self {
        Self {
            width: 0,
            color: Color::rgb(0, 0, 0),
            style: BorderStyle::Solid,
        }
    }
}

/// Border style enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BorderStyle {
    /// Solid border
    Solid,
    /// Dashed border
    Dashed,
    /// Dotted border
    Dotted,
    /// Double border
    Double,
}
