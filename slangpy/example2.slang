// example.slang
import "slangpy";

struct Pixel
{
    float r;
    float g;
    float b;
};

// Add two pixels together
void add(
    int2 index,
    ITensor<Pixel, 2> a,
    ITensor<Pixel, 2> b,
    IRWTensor<Pixel, 2> result)
{
    // Load pixels from input tensors
    Pixel pixel_a = a[index];
    Pixel pixel_b = b[index];

    // Add the pixels
    Pixel pixel_result;
    pixel_result.r = pixel_a.r + pixel_b.r;
    pixel_result.g = pixel_a.g + pixel_b.g;
    pixel_result.b = pixel_a.b + pixel_b.b;

    // Store the result
    result[index] = pixel_result;
}