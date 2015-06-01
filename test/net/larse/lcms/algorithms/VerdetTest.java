package net.larse.lcms.algorithms;

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import net.larse.lcms.helper.Raster;
import org.gdal.gdal.Dataset;
import org.gdal.gdal.Driver;
import org.gdal.gdal.gdal;
import org.gdal.gdalconst.gdalconstConstants;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by yang on 5/2/15.
 */
public class VerdetTest {
  double[] x;
  double[] expected;
  @Before
  public void setUp() throws Exception {
    x = new double[] {0.00074769,  0.00209049, -0.00343328,  0.00108339,  0.0230869 ,
        -0.01185626, -0.00041199,  0.00291447, -0.22185092, -0.15471122,
        -0.19047837, -0.17409018, -0.1922179 , -0.17494468, -0.2184329 ,
        -0.14006256, -0.17701991, -0.18001068, -0.16420233, -0.17152667,
        -0.17253377, -0.15910582, -0.16334783, -0.1172961 , -0.14055085,
        -0.14149691, -0.13392843, -0.10453956, -0.13798733, -0.1374075};

    expected = new double[] {0, -0.00014025,-0.00014025,-0.00014025,-0.00014025,-0.00014025,
        -0.00014025,-0.00014025,-0.00014025,-0.00014025,-0.00014025,-0.00014025,-0.00014025,
        -0.00014025,-0.00014025,-0.00014025,-0.00014025,-0.00014025,-0.00014025,-0.00014025,
        -0.00014025,-0.00014025,-0.00014025,-0.00014025,-0.00014025,-0.00014025,-0.00014025,
        -0.00014025,-0.00014025,-0.00014025};
  }

  @After
  public void tearDown() throws Exception {

  }

  @Ignore
  @Test
  public void testGetResult() throws Exception {
    Verdet verdet = new Verdet();
    double[] score = verdet.getResult(x);

    //compare score with expected
  }

  @Test
  public void testSpatialRun() {
    //TODO: reorganize the code to improve efficiency
    //Note: In this test code, the output is one band for each year, which is different from the matlab verdet output
    // where outputs are consecutive year-pairs, and the first year has not output.

    String srcStackFile = "/Users/yang/CloudStorage/GoogleDrive/CodeWorkspaces/GAE/lcms/test/test_verdet/NDMI_Z_Stack.tif";
    String scoreFile = "/Users/yang/Downloads/junk/test_verdet.tif";
    String expectedFile = "/Users/yang/CloudStorage/GoogleDrive/CodeWorkspaces/GAE/lcms/test/test_verdet/verdet_output_score.tif";

    int failedPixel = 0;

    Raster stack = new Raster(srcStackFile);
    Raster expected = new Raster(expectedFile);

    Verdet verdet = new Verdet();

    double mapX = stack.ulx;
    double mapY = stack.uly;
    int xsize = stack.ds.getRasterXSize();
    int ysize = stack.ds.getRasterYSize();
    int nYears = stack.bands.length;

    int[] scores = new int[xsize * ysize * nYears];

    //used for compare with expected values
    double[] comparedScore = new double[nYears-1];

    for (int y = 0; y < ysize; y++) {
      for (int x = 0; x < xsize; x++) {
        double mapx = mapX + x * stack.pixelX;
        double mapy = mapY + y * stack.pixelY;

        double[] values = stack.Read(mapx, mapy, 1, 1);
        double[] expectedResults = expected.Read(mapx, mapy, 1, 1);

        for (int k=0; k < nYears; k++) {
          values[k] = values[k] / 65535.0 * 2.0 - 1.0;
        }
        double[] pixelScores = verdet.getResult(values);

        for (int z = 0; z < nYears; z++) {
          int offset = xsize * ysize * z + y * xsize + x;
          scores[offset] = (int)((pixelScores[z]+0.092938733186154518) * 65535.0 / 0.10414045959466088);
          if (z>0) {
            comparedScore[z-1] = scores[offset];
          }
        }

        if (!Arrays.equals(expectedResults, comparedScore)) {
          System.out.println(String.format("failed at (%d, %d)", x, y));
          failedPixel++;
        }
      }
    }

    //TODO: wrapp this in Raster class. Raster class is not a complete implementation.
    Driver driver = gdal.GetDriverByName("GTiff");
    Dataset rds = driver.Create(scoreFile, xsize, ysize, nYears, gdalconstConstants.GDT_Int32);
    rds.SetProjection(stack.ds.GetProjection());
    double[] geo = stack.ds.GetGeoTransform();
    geo[0] = mapX;
    geo[3] = mapY;

    rds.SetGeoTransform(geo);
    rds.WriteRaster(0, 0, xsize, ysize, xsize, ysize, gdalconstConstants.GDT_Int32, scores, stack.bands);
    rds.delete();

    assertTrue(String.format("Failed pixels: %d", failedPixel), failedPixel==0);

  }
}