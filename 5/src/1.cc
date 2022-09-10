/*
Rendering the complex numbers c such that the f(z) = z^2+c map has a stable
orbit, i.e. the Mandelbrot set. Coloring can be done in multiple ways. The image
can be written to png right away, or to netcdf in order to plot in matplotlib or
or paraview or whatever.
*/

#include "1.hh"

int main()
{
    using namespace Mandelbrot;

    {
    constexpr range_t range {{-2, 1}, {-1, 1}}; // -2 to 1, -i to i
    constexpr int2 res {3840, 2560};
    constexpr size_t max_i = 1e3;
    constexpr int color_type = PNG_COLOR_TYPE_RGB;
    constexpr int bytes = 3;
    constexpr int bit_depth = 8;
    constexpr size_t channels = 3;

    auto *image = new png_byte[channels*res.x*res.y];
    auto map = color_map_1<png_byte>;
    auto assigner = color_assigner<png_byte, channels>;
    render<max_i, channels>(image, range, res, map, assigner);
    write_png<color_type, bytes, bit_depth>("a.png", image, res);
    delete[] image;
    }

    {
    constexpr double2 c {0.001643721971153, -0.822467633298876};
    constexpr double r = 1e-11;
    constexpr range_t range {{c.x-r, c.x+r}, {c.y-r, c.y+r}};
    constexpr int2 res {1280, 1280};
    constexpr size_t max_i = 1e4;

    auto *image = new float[res.x*res.y];
    auto map = color_map_2<float>;
    auto assigner = color_assigner<float>;
    render<max_i>(image, range, res, map, assigner);
    write_nc<float, NC_FLOAT>("b.nc", image, res);
    delete[] image;
    }
}
