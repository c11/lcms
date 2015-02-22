/*
 * Copyright (c) 2015 Zhiqiang Yang.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package net.larse.lcms.helper;

import org.gdal.gdal.gdal;
import org.gdal.gdal.Dataset;
import org.gdal.gdalconst.gdalconstConstants;



/**
 * Created by yang on 2/8/15.
 */
public class Raster {
  private String fileName = "";

  public Dataset ds = null;

  public double ulx = Double.NEGATIVE_INFINITY;
  public double uly = Double.NEGATIVE_INFINITY;
  public double pixelX = 0;
  public double pixelY = 0;
  public int[] bands;

  public Raster(String fileName) {
    this(fileName, false);
  }

  public Raster(String fileName, boolean create) {
    gdal.AllRegister();

    this.fileName = fileName;

    if (!create) {
      ds = gdal.Open(this.fileName, gdalconstConstants.GA_ReadOnly);

      double[] geo = ds.GetGeoTransform();
      ulx = geo[0];
      uly = geo[3];
      pixelX = geo[1];
      pixelY = geo[5];

      bands = new int[ds.getRasterCount()];
      for (int i = 0; i < bands.length; i++) {
        bands[i] = i + 1;
      }
    }
  }

  /**
   * Read a box with specified dimensions using map coordinates
   * @param x upper left corner X
   * @param y upper left coorner Y
   * @param xsize box width
   * @param ysize box height
   * @return double[]
   */
  public double[] Read(double x, double y, int xsize, int ysize) {
    //TODO: add assertion
    int xoffset = (int)((x-ulx)/pixelX);
    int yoffset = (int)((y-uly)/pixelY);

    double[] values = new double[xsize * ysize * bands.length];

    int error = ds.ReadRaster(xoffset, yoffset, xsize, ysize, xsize, ysize, gdalconstConstants.GDT_Float64, values, bands);

    return values;
  }

}
