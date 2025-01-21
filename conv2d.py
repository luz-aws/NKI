import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

import sys
"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]


    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_pmax
    n_tiles_c_out = out_channels // c_pmax
    out_chunks = 2 # may need to set as pool_size
    n_out_chunks = (out_height + out_chunks - 1) // out_chunks
    chunk_height = out_chunks + filter_height - 1

    #- load in the weights into an SBUF array of shape:
    W = W.reshape((n_tiles_c_out, c_pmax, n_tiles_c_in, c_pmax, filter_height, filter_width))

    W_sbuf = nl.ndarray(
        shape=(n_tiles_c_out, nl.par_dim(c_pmax), n_tiles_c_in, c_pmax, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf
    )


    for c_out_tile in nl.affine_range(n_tiles_c_out):
        W_sbuf[c_out_tile] = nl.load(W[c_out_tile])
    
    
    weight_copy = nl.ndarray(
        shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_pmax), c_pmax),
        dtype=W.dtype,
        buffer=nl.sbuf
    )
    w = nl.ndarray(
        shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_pmax), c_pmax),
        dtype=W.dtype,
        buffer=nl.sbuf
    )

    for c_out_tile in nl.affine_range(n_tiles_c_out):
        for c_in_tile in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    weight_copy[i, j, c_out_tile, c_in_tile, :, :] = nl.copy(W_sbuf[c_out_tile, :, c_in_tile, :, i, j], dtype = W.dtype)
                    w[i, j, c_out_tile, c_in_tile] = nisa.nc_transpose(weight_copy[i, j, c_out_tile, c_in_tile])
        
    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for n in nl.affine_range(n_out_chunks):
            x = nl.ndarray(
                shape=(n_tiles_c_in, nl.par_dim(c_pmax), chunk_height, input_width),
                dtype=X.dtype,
                buffer=nl.sbuf
            )

            for c_in_tile in nl.affine_range(n_tiles_c_in):
                x[c_in_tile] = nl.load(X[b, 
                                       c_in_tile*c_pmax : (c_in_tile + 1)*c_pmax, 
                                       n*out_chunks : (n*out_chunks) + chunk_height, 
                                       :])
            
            for c_out_tile in nl.affine_range(n_tiles_c_out):
                conv_output = nl.ndarray(
                    shape=(nl.par_dim(c_pmax), out_chunks, out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf
                )
                bias_sbuf = nl.ndarray(
                    shape=(nl.par_dim(c_pmax),),
                    dtype=bias.dtype,
                    buffer=nl.sbuf
                )
                bias_sbuf = nl.load(bias[c_out_tile * c_pmax : (c_out_tile + 1) * c_pmax],)
                for output_row in nl.affine_range(out_chunks):
                    #- assign space in PSUM to store output row
                    output_row_psum = nl.zeros((nl.par_dim(c_pmax), out_width), nl.float32, buffer=nl.psum)
                    for filter_row in nl.affine_range(filter_height):
                        for filter_col in nl.affine_range(filter_width):
                            for c_in_tile in nl.affine_range(n_tiles_c_in):
                                output_row_psum += nl.matmul(
                                    w[filter_row, filter_col, c_out_tile, c_in_tile, :, :],
                                    x[c_in_tile, :, output_row + filter_row, filter_col:filter_col + out_width],
                                    transpose_x=True
                                )

                    #- copy stuff from PSUM back to SBUF
                    b_output_row = nisa.tensor_scalar(output_row_psum, np.add, bias_sbuf)
                    conv_output[:,output_row,:] = nl.copy(b_output_row, dtype=X.dtype)
                
                #- copy stuff from SBUF back to HBM
                nl.store(X_out[b, 
                         c_out_tile * c_pmax : (c_out_tile + 1) * c_pmax, 
                         n*out_chunks:(n+1)*out_chunks, 
                         :], 
                         value=conv_output)
    return X_out
        
        #raise RuntimeError("Please fill your implementation of computing convolution"
         #                  " of X[b] with the weights W and bias b, followed by a"
         #                  " maxpool and store the result in X_out[b]")

    return X_out

