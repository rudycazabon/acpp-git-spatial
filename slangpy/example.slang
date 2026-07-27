// Currently, to use custom types with SlangPy, they need to be explicitly imported.
import "slangpy";

// example.slang
struct Pixel
{
    float r;
    float g;
    float b;
};

// Add two pixels together
Pixel add(Pixel a, Pixel b)
{
    Pixel result;
    result.r = a.r + b.r;
    result.g = a.g + b.g;
    result.b = a.b + b.b;
    return result;
}