/*
Render 3D Lyanupov fractals of arbitrary iteration sequence. Data is written
to a compressed netcdf file. So this is a cubic grid file; not 3D in the sense
of having a value at every 2D point, but 3D as in the sense of having a value at
every 3D point. This can't be rendered easily. Need to use opacity to cut off
negative values or whatever in a 3D volumetric render. This requires a true 3D
volumetric renderer capable of working with 3D volumetric data, like Paraview.
The netcdf files can also easily be opened in Paraview.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <netcdf.h>
#include <netcdf_filter.h>

int main()
{
    // 3D resolution
    constexpr size_t res_A = 512;
    constexpr size_t res_B = 512;
    constexpr size_t res_C = 512;
    // 3D ABC range
    constexpr float lim_A[2] {3, 4};
    constexpr float lim_B[2] {3, 4};
    constexpr float lim_C[2] {3, 4};
    // number of sequence iterations, e.g. ABCABC is 2 sequence iterations
    constexpr size_t n_seq_i = 1000;
    // initial condition in logistic map
    constexpr float x0 = .1;
    // iteration sequence length, e.g. ABC is a length 3 sequence
    constexpr size_t seq_l = 5;
    // filename
    constexpr char filename[] = "Lyapunov ABABC.nc";

    // define the sequence
    enum r_t {A, B, C};
    constexpr r_t seq[seq_l] {A, B, A, B, C};

    // allocating and zero initializing Lyapunov exponents
    auto *l = new float[res_A*res_B*res_C] {};

    #pragma omp parallel for
    for (size_t a = 0; a < res_A; a++)
        for (size_t b = 0; b < res_B; b++)
            for (size_t c = 0; c < res_C; c++)
            {   size_t j = res_C*(res_B*a+b)+c; // 3D tensor index
                float x = x0;
                for (size_t i = 0; i < n_seq_i; i++)
                {   // pixel to ABC coordinate conversion
                    float r[3] {lim_A[0]+(lim_A[1]-lim_A[0])/res_A*a,
                                lim_B[0]+(lim_B[1]-lim_B[0])/res_B*b,
                                lim_C[0]+(lim_C[1]-lim_C[0])/res_C*c};
                    // iteration sequence
                    for (size_t k = 0; k < seq_l; k++)
                    {   x = r[seq[k]]*x*(1-x);
                        l[j] += log(abs(r[seq[k]]*(1-2*x)));
                    }
                }
            }

    // Lyapunov exponent needs this 'normalization'
    #pragma omp parallel for
    for (size_t a = 0; a < res_A; a++)
        for (size_t b = 0; b < res_B; b++)
            for (size_t c = 0; c < res_C; c++)
                l[a*res_B+b] /= n_seq_i*seq_l;

    // check for netcdf errors
    auto check = [](int retval)
    {   if (retval != NC_NOERR)
        {   printf("nc error: %s\n", nc_strerror(retval));
            exit(EXIT_FAILURE);
        }
    };

    // szip compressed 3D netcdf write
    int id_nc, id_dims[3], id_var;
    constexpr size_t chunks[2] {32, 32};
    check(nc_create(filename, NC_CLOBBER|NC_NETCDF4, &id_nc));
    check(nc_set_fill(id_nc, NC_NOFILL, NULL));
    check(nc_def_dim(id_nc, "A", res_A, id_dims));
    check(nc_def_dim(id_nc, "B", res_B, id_dims+1));
    check(nc_def_dim(id_nc, "C", res_B, id_dims+2));
    check(nc_def_var(id_nc, "Lyapunov ABABC", NC_FLOAT, 3, id_dims, &id_var));
    check(nc_def_var_szip(id_nc, 0, chunks[0], chunks[1]));
    check(nc_enddef(id_nc));
    check(nc_put_var(id_nc, id_var, l));
    check(nc_close(id_nc));

    delete[] l;
}
