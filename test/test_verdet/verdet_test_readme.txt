To match with original implementation, these test files are scaled to int16. To use them, the pixel values need to be scaled back to the actual value using the following equations.

NDMI_Z_Stack.tif: this is a 30 bands image, each band represents a year.

         actual_value = pixel_value / 65535.0 * 2.0 - 1.0

verdet_output_score: this is the output images with 29 bands
     
         actual_value = pixel_value / 65535.0 * 0.10414045959466088 -0.092938733186154518