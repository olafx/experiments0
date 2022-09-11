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

    constexpr range_t range {{-2, 1}, {-1, 1}}; // -2 to 1, -i to i
    constexpr size_t2 res {3840, 2560};
    constexpr size_t max_i = 1e3;

    {
    constexpr int bytes = 3;
    constexpr int bit_depth = 8;
    constexpr size_t channels = 3;

    auto *image = new png_byte[channels*res.x*res.y];
    auto mapper = color_map_1<png_byte, max_i>;
    auto assigner = color_assigner<png_byte, channels>;
    render<max_i, channels>(image, range, res, mapper, assigner);
    write_png<PNG_COLOR_TYPE_RGB, bytes, bit_depth>("out/a.png", image, res);
    delete[] image;
    }

    {
    auto *image = new float[res.x*res.y];
    auto mapper = color_map_3<float, max_i>;
    auto assigner = color_assigner<float>;
    render<max_i>(image, range, res, mapper, assigner);
    write_nc<float, NC_FLOAT>("out/b.nc", image, res);
    delete[] image;
    }
}
