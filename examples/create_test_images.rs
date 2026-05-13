use image::{DynamicImage, ImageBuffer, Rgb};

fn main() {
    // Create test-images directory if it doesn't exist
    std::fs::create_dir_all("test-images").unwrap();

    // Create a simple test image with text patterns using ImageBuffer
    let width = 300u32;
    let height = 100u32;

    // Create a white image buffer
    let rgb_img = ImageBuffer::from_fn(width, height, |x, y| {
        // Default to white
        Rgb([255, 255, 255])
    });

    // Create a DynamicImage from the buffer
    let mut img = DynamicImage::ImageRgb8(rgb_img);

    // Save the blank image
    img.save("test-images/blank.png").unwrap();
    println!("Created test-images/blank.png");

    // Create a simple single character image
    let single_rgb = ImageBuffer::from_fn(50, 50, |x, y| {
        if x >= 15 && x <= 25 && y >= 10 && y <= 40 {
            Rgb([0, 0, 0]) // Black rectangle to simulate a character
        } else {
            Rgb([255, 255, 255]) // White background
        }
    });
    let single_img = DynamicImage::ImageRgb8(single_rgb);
    single_img.save("test-images/single_char.png").unwrap();
    println!("Created test-images/single_char.png");

    // Create a simple text image
    let text_rgb = ImageBuffer::from_fn(200, 50, |x, y| {
        // Create multiple black rectangles to simulate text
        if x >= 10 && x <= 20 && y >= 10 && y <= 30 {
            Rgb([0, 0, 0]) // First character
        } else if x >= 30 && x <= 40 && y >= 10 && y <= 30 {
            Rgb([0, 0, 0]) // Second character
        } else {
            Rgb([255, 255, 255]) // White background
        }
    });
    let text_img = DynamicImage::ImageRgb8(text_rgb);
    text_img.save("test-images/simple_text.png").unwrap();
    println!("Created test-images/simple_text.png");

    // Create a noisy text image
    let noisy_rgb = ImageBuffer::from_fn(200, 100, |x, y| {
        // Create some noise
        if (x + y) % 7 == 0 {
            let gray = ((x * y) % 128) as u8;
            Rgb([gray, gray, gray])
        } else if x >= 50 && x <= 60 && y >= 30 && y <= 50 {
            Rgb([0, 0, 0]) // First character
        } else if x >= 70 && x <= 80 && y >= 30 && y <= 50 {
            Rgb([0, 0, 0]) // Second character
        } else {
            Rgb([255, 255, 255]) // White background
        }
    });
    let noisy_img = DynamicImage::ImageRgb8(noisy_rgb);
    noisy_img.save("test-images/noisy_text.png").unwrap();
    println!("Created test-images/noisy_text.png");

    println!("All test images created successfully in test-images/");
}
