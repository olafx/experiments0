#include <cstdio>
#include <cmath>
#include <hdf5.h>

int main(int argc, char **argv)
{
    // HDF5 settings
    char *name_file_ic = argv[1];
    char *name_file    = argv[2];

    // physics settings
    size_t N;   // number of timesteps
    size_t N_s; // store every this many steps
    double dt;  // timestep
    double e2;  // softening epsilon squared
    sscanf(argv[3], "%zu", &N);
    sscanf(argv[4], "%zu", &N_s);
    sscanf(argv[5], "%lf", &dt);
    sscanf(argv[6], "%lf", &e2);

/******************************************************************************/

    // shared HDF5 handles
    hid_t dataspace, cparms, file, filespace;
    herr_t status; // not checking status
    // unique HDF5 handles
    hid_t ic_dataset;
    hid_t pv_dataset, pv_memoryspace;
    hid_t  t_dataset,  t_memoryspace;
    // HDF5 dims
    hsize_t ic_dims[3];
    hsize_t pv_dims[4], pv_dims_max[4], pv_dims_chunk[4], pv_offset[4];
    hsize_t  t_dims[1],  t_dims_max[1],  t_dims_chunk[1],  t_offset[1];

    // physics data
    double *data;
    size_t n; // # objects

    // physics convenience pointers
    double *data_pv;
    double *data_p;
    double *data_v;
    double *data_t;

/******************************************************************************/

    // reading initial condition size
    file       = H5Fopen(name_file_ic, H5F_ACC_RDONLY, H5P_DEFAULT);
    ic_dataset = H5Dopen2(file, "pos,vel", H5P_DEFAULT);
    dataspace  = H5Dget_space(ic_dataset);
    status     = H5Sget_simple_extent_dims(dataspace, ic_dims, NULL);
    n = ic_dims[1];

    // physics data allocation
    data = new double[6*n+1];

    // physics convenience pointers assignment
    data_pv = data;
    data_p  = data;
    data_v  = data+3*n;
    data_t  = data+6*n;

    // reading initial condition
    status = H5Dread(ic_dataset, H5T_NATIVE_DOUBLE, dataspace, dataspace,
                 H5P_DEFAULT, data_pv);

    // closing initial condition file
    status = H5Dclose(ic_dataset);
    status = H5Sclose(dataspace);
    status = H5Fclose(file);

/******************************************************************************/

    // create new file, overriding
    file = H5Fcreate(name_file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // dims setup
    pv_dims[0]       = N / N_s + 1;
    pv_dims[1]       = 2;
    pv_dims[2]       = n;
    pv_dims[3]       = 3;
    pv_dims_max[0]   = H5S_UNLIMITED;
    pv_dims_max[1]   = 2;
    pv_dims_max[2]   = n;
    pv_dims_max[3]   = 3;
    pv_dims_chunk[0] = 1;
    pv_dims_chunk[1] = 2;
    pv_dims_chunk[2] = n;
    pv_dims_chunk[3] = 3;
     t_dims[0]       = N / N_s + 1;
     t_dims_max[0]   = H5S_UNLIMITED;
     t_dims_chunk[0] = 1;

    // create extendible, chunked dataset for positions and velocities
    constexpr double fill = 0;
    dataspace  = H5Screate_simple(4, pv_dims, pv_dims_max);
    cparms     = H5Pcreate(H5P_DATASET_CREATE);
    status     = H5Pset_chunk(cparms, 4, pv_dims_chunk);
    status     = H5Pset_fill_value(cparms, H5T_NATIVE_DOUBLE, &fill);
    pv_dataset = H5Dcreate2(file, "pos,vel", H5T_NATIVE_DOUBLE, dataspace,
                     H5P_DEFAULT, cparms, H5P_DEFAULT);

    // create extendible, chunked dataset for time
    dataspace = H5Screate_simple(1, t_dims, t_dims_max);
    cparms    = H5Pcreate(H5P_DATASET_CREATE);
    status    = H5Pset_chunk(cparms, 1, t_dims_chunk);
    status    = H5Pset_fill_value(cparms, H5T_NATIVE_DOUBLE, &fill);
    t_dataset = H5Dcreate2(file, "time", H5T_NATIVE_DOUBLE, dataspace,
                    H5P_DEFAULT, cparms, H5P_DEFAULT);

    // create offsets for writing
    pv_offset[0] = 0;
    pv_offset[1] = 0;
    pv_offset[2] = 0;
    pv_offset[3] = 0;
     t_offset[0] = 0;

    // create memoryspaces for writing
    pv_memoryspace = H5Screate_simple(4, pv_dims_chunk, NULL);
     t_memoryspace = H5Screate_simple(1,  t_dims_chunk, NULL);

    // lambda for writing
    auto write = [&](size_t t)
    {   // write positions and velocities
        filespace = H5Dget_space(pv_dataset);
        status    = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, pv_offset,
                        NULL, pv_dims_chunk, NULL);
        status    = H5Dwrite(pv_dataset, H5T_NATIVE_DOUBLE, pv_memoryspace,
                        filespace, H5P_DEFAULT, data_pv);
        pv_offset[0]++;
        // write time
        *data_t = t*dt;
        filespace = H5Dget_space(t_dataset);
        status    = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, t_offset,
                        NULL, t_dims_chunk, NULL);
        status    = H5Dwrite(t_dataset, H5T_NATIVE_DOUBLE, t_memoryspace,
                        filespace, H5P_DEFAULT, data_t);
        t_offset[0]++;
        printf("%zu/%zu\n", t, N);
    };

    // write initial condition
    status = H5Dset_extent(pv_dataset, pv_dims);
    status = H5Dset_extent(t_dataset, t_dims);
    write(0);

/******************************************************************************/

    // 1/(2-2^(1/3))
    constexpr double a = 1.351207191959657634047687808971460;

    auto pos = [&](const double x)
    {   for (size_t i = 0; i < n; i++)
        {   data_p[3*i  ] += x*data_v[3*i  ]*dt;
            data_p[3*i+1] += x*data_v[3*i+1]*dt;
            data_p[3*i+2] += x*data_v[3*i+2]*dt;
        }
    };
    auto vel = [&](const double y)
    {   for (size_t i = 0; i < n; i++)
        {   double a1 = 0;
            double a2 = 0;
            double a3 = 0;
            for (size_t j = 0; j < n; j++)
            {   if (i != j)
                {   const double b1 = data_p[3*j  ]-data_p[3*i  ];
                    const double b2 = data_p[3*j+1]-data_p[3*i+1];
                    const double b3 = data_p[3*j+2]-data_p[3*i+2];
                    double c = 1/sqrt(b1*b1+b2*b2+b3*b3+e2);
                    c = c*c*c;
                    a1 += c*b1;
                    a2 += c*b2;
                    a3 += c*b3;
                }
            }
            data_v[3*i  ] += y*a1*dt;
            data_v[3*i+1] += y*a2*dt;
            data_v[3*i+2] += y*a3*dt;
        }
    };

    // time steps
    for (size_t t = 1; t <= N; t++)
    {
        pos(.5*a);
        vel(a);
        pos(.5*(1-a));
        vel(1-2*a);
        pos(.5*(1-a));
        vel(a);
        pos(.5*a);

        // process
        if (t % N_s == 0)
            write(t);
    }

/******************************************************************************/

    // HDF5 closing
    H5Dclose(pv_dataset);
    H5Sclose(pv_memoryspace);
    H5Dclose(t_dataset);
    H5Sclose(t_memoryspace);
    H5Sclose(dataspace);
    H5Sclose(filespace);
    H5Pclose(cparms);
    H5Fclose(file);

    // physics data deallocation
    delete[] data_pv;
}
