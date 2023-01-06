#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include <png.h>
#include <netcdf.h>
#include <netcdf_filter.h>

namespace Mandelbrot
{

template <typename T>
struct vec2 { T x, y; };

using double2 = vec2<double>;
using size_t2 = vec2<size_t>;
using range_t = vec2<double[2]>;

template <int png_color_type, int png_bytes, int png_bit_depth>
void write_png(const char *const filename, png_byte *const image, const size_t2 res)
{
    constexpr bool debug_png = false;

    auto check = [](auto *ret_pointer)
    {   if (ret_pointer == NULL)
        {   if constexpr (debug_png)
                puts("png error");
            exit(EXIT_FAILURE);
        }
    };

    FILE *fp = fopen(filename, "wb");
    check(fp);

    png_struct *writer = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    check(writer);
    setjmp(png_jmpbuf(writer));
    png_init_io(writer, fp);

    png_info *info = png_create_info_struct(writer);
    check(info);
    png_set_IHDR(writer, info, res.x, res.y, png_bit_depth,
        png_color_type, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(writer, info);

    png_byte **image_rows = new png_byte *[res.y];
    for (size_t i = 0; i < res.y; i++)
        image_rows[i] = image+res.x*png_bytes*i;

    png_write_image(writer, image_rows);

    png_write_end(writer, NULL);
    delete[] image_rows;
    fclose(fp);
}

template <typename A, nc_type type>
void write_nc(const char *const filename, A *const image, const size_t2 res)
{
    constexpr bool debug_nc = false;

    auto check = [](int retval)
    {   if (retval != NC_NOERR)
        {   if constexpr (debug_nc)
                printf("nc error: %s\n", nc_strerror(retval));
            exit(EXIT_FAILURE);
        }
    };

    int id_nc, id_dims[2], id_var;
    constexpr size_t chunks[2] {32, 32}; // max szip chunk size
    check(nc_create(filename, NC_CLOBBER|NC_NETCDF4, &id_nc));
    check(nc_set_fill(id_nc, NC_NOFILL, NULL));
    check(nc_def_dim(id_nc, "imag", res.y, id_dims));
    check(nc_def_dim(id_nc, "real", res.x, id_dims+1));
    check(nc_def_var(id_nc, "Mandelbrot", type, 2, id_dims, &id_var));
    check(nc_def_var_szip(id_nc, 0, chunks[0], chunks[1]));
    check(nc_enddef(id_nc));
    check(nc_put_var(id_nc, id_var, image));
    check(nc_close(id_nc));
}

// example of RGB coloring
template <typename A, size_t max_i>
void color_map_1(const double2 z, const size_t i, A *const color)
{   double log1p_i = log1p(i);
    static double max_log1p_i = log1p(max_i); // log1p isn't marked constexpr
    color[0] = 0xFF;
    color[1] = log1p_i/max_log1p_i*0xFF;
    color[2] = 0x80;
}

// example of gray coloring
template <typename A>
void color_map_2(const double2 z, const size_t i, A *const color)
{   *color = i;
}

// example of continuous gray coloring
template <typename A, size_t max_i>
void color_map_3(const double2 z, const size_t i, A *const color)
{   constexpr double r = 2;
    *color = i == max_i ? max_i : i-log2(.5*log(z.x*z.x+z.y*z.y)/log(r));
}

template <typename A, size_t channels = 1>
void color_assigner(A *const image, const size_t2 res, const size_t2 p, const A *const color)
{
    for (size_t j = 0; j < channels; j++)
    {   image[channels*(p.y*res.x+p.x)+j] = color[j];
    }
}

template <size_t max_i, size_t channels = 1, typename image_type, typename A, typename B>
void render(image_type *image, const range_t range, const size_t2 res, A &color_mapper, B &color_assigner)
{
    auto p_to_c = [&](const size_t2 p)
    {   return double2 {range.x[0]+(range.x[1]-range.x[0])/res.x*p.x,
                        range.y[0]+(range.y[1]-range.y[0])/res.y*p.y};
    };

    constexpr double r2 = 4;

    #pragma omp parallel for
    for (size_t x = 0; x < res.x; x++)
        for (size_t y = 0; y < res.y; y++)
        {   double2 c = p_to_c({x, y});
            double2 z = {0, 0};
            size_t i;
            for (i = 0; i < max_i && z.x*z.x+z.y*z.y <= r2; i++)
            {   double a = z.x*z.x-z.y*z.y+c.x;
                z.y = 2*z.x*z.y+c.y;
                z.x = a;
            }
            image_type color[channels];
            color_mapper(z, i, color);
            color_assigner(image, res, {x, y}, color);
        }
}

}
