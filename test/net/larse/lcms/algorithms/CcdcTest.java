 package net.larse.lcms.algorithms;
//package com.google.earthengine.examples.experimental;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

//import com.google.earthengine.examples.experimental.Ccdc.CcdcFit;
//import com.google.testing.util.TestUtil;
import net.larse.lcms.algorithms.Ccdc.CcdcFit;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.lang.reflect.Field;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.ejml.data.DenseMatrix64F;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Created by yang on 6/9/17. */
@RunWith(JUnit4.class)
public class CcdcTest {
  private static final int DAY1970 = 719528;

  private static final String PACKAGE_NAME = CcdcTest.class.getPackage().getName();

  //NAFD samples
//  private static final String FILE_PREFIX =
//     "/Users/yang/CloudStorage/GoogleDrive/CodeWorkspaces/TTU/GEE-Validation/Test/testdata/";
//  private static final String GOLDEN_DATA_FILE = FILE_PREFIX + "lcmap_coefs_matlab.csv";

  //global sample
  private static final String FILE_PREFIX =
          "./lcms/test/test_ccdc/";
  private static final String GOLDEN_DATA_FILE = FILE_PREFIX + "global_stratified_ccdc_coefs_matlab.csv";


  // How many points to test.
  private static final int TEST_COUNT = 0;


  @Test
  public void testGolderData() throws Exception {
    // Run CCDC on all the input points.
    Map<String, List<CcdcFit>> result = new HashMap<>();

    String dat_dir = FILE_PREFIX + "points";
    String[] csvs = new File(dat_dir).list((dir, name) -> name.endsWith(".csv"));
    int count = 0;
    if (csvs != null) {
      for (String f : csvs) {
        System.out.format("Processing %s\n", f);
        System.out.flush();
        DenseMatrix64F matrix = readTimeSeries(dat_dir + '/' + f);
        String[] items = f.replace(".csv", "").split("_");
        result.put(items[1], runCcdc(matrix));
        if (TEST_COUNT > 0 && ++count == TEST_COUNT) {
          break;
        }
      }
    }

    // Verify results against golden data.
    Map<String, List<CcdcFit>> expected = readTestData();

    StringBuffer buf = new StringBuffer();

    for (String key : result.keySet()) {
      System.out.format("Comparing %s\n", key);

      List<CcdcFit> want = expected.get(key);
      List<CcdcFit> have = result.get(key);

      if (want.size() != have.size()) {
        buf.append(String.format(
            "Object mismatch.  Plot: %s.  Expected %d rows, found %d.\n", key, want.size(),
            have.size()));
        continue;
      }
      have.sort(Comparator.comparing((CcdcFit fit) -> fit.tStart));
      for (int i = 0; i < have.size(); i++) {
        StringBuffer b = new StringBuffer();
        if (!nearlyEqual(want.get(i), have.get(i), 0.1, b)) {
          buf.append("Object mismatch.  Plot: " + key + " row: " + i + "\n" + b.toString());
        }
      }
    }
    if (buf.length() > 0) {
      fail(buf.toString());
    }
  }

  private Map<String, List<CcdcFit>> readTestData() throws Exception {
    Map<String, List<CcdcFit>> map = new HashMap<>();
    BufferedReader reader;
    reader = new BufferedReader(new FileReader(GOLDEN_DATA_FILE));

    String line = reader.readLine(); // ignore header line
    while ((line = reader.readLine()) != null) {
      String[] items = line.trim().replace("NA", "").replace(",,", ",").split(",");

      CcdcFit fit = parseTestDataLine(line);
      List<CcdcFit> list = map.getOrDefault(items[0], new ArrayList<>());
      if (fit.tStart > 0) {
        list.add(fit);
      }
      map.put(items[0], list);
    }

    return map;
  }

  private CcdcFit parseTestDataLine(String line) {
    String[] items = line.trim().replace("NA", "").replace(",,", ",").split(",");

    // pos,tStart,tEnd,tBreak,numObs,changeProb,category,coef[7][8],rmse[7],magnitude[7]
    CcdcFit fit = new CcdcFit();
    fit.tStart = Integer.parseInt(items[1]);
    fit.tEnd = Integer.parseInt(items[2]);
    fit.tBreak = Integer.parseInt(items[3]);
    fit.numObs = Integer.parseInt(items[4]);
    fit.changeProb = Double.parseDouble(items[5]);
    fit.category = Integer.parseInt(items[6]);
    fit.coefs = new double[7][8];
    fit.rmse = new double[7];
    fit.magnitude = new double[7];
    for (int b = 0; b < 7; b++) {
      for (int p = 0; p < 8; p++) {
        fit.coefs[b][p] = Double.parseDouble(items[7 + b * 8 + p]);
      }
      fit.rmse[b] = Double.parseDouble(items[7 + 56 + b]);
      fit.magnitude[b] = Double.parseDouble(items[7 + 56 + 7 + b]);
    }
    return fit;
  }

  private List<CcdcFit> runCcdc(DenseMatrix64F matrix) throws Exception {
    double[] juldays = new double[matrix.numRows];
    double[][] refls = new double[matrix.numRows][8];

    for (int i = 0; i < matrix.numRows; i++) {
      // convert year-day to matlab datenum
      juldays[i] = LocalDate.of((int) matrix.get(i, 0), 1, 1).toEpochDay()
          + matrix.get(i, 1) + DAY1970;
      for (int j = 2; j < matrix.numCols; j++) {
        refls[i][j - 2] = matrix.get(i, j);
      }
    }
    return new Ccdc().getResult(refls, juldays);
//    return new LcmapCcdc().getResult(refls, juldays);
  }

  private DenseMatrix64F readTimeSeries(String filename) throws Exception {
    DoubleArrayList values = new DoubleArrayList();
    int nrow = 0;

    BufferedReader reader = new BufferedReader(new FileReader(filename));
    String line = reader.readLine(); // ignore header line
    while ((line = reader.readLine()) != null) {
      // sensor,pid,tsa,plotid,yr,day,b1,b2,b3,b4,b5,b7,b6,fmask
      String[] items = line.trim().split(",");
      assertEquals(items.length, 14);

      for (int i = 4; i < items.length; i++) {
        values.add(Double.parseDouble(items[i]));
      }
      nrow++;

      if (nrow * 10 != values.size()) {
        System.out.println("something wrong");
      }
    }
    reader.close();
    return new DenseMatrix64F(nrow, 10, true, values.toDoubleArray());
  }

  /** Test that two fits are nearly equal, to within the specified tolerance. */
  private boolean nearlyEqual(CcdcFit expected, CcdcFit actual, double tolerance, StringBuffer buf)
      throws Exception {
    compare(expected, actual, "tStart", 0, buf);
    compare(expected, actual, "tEnd", 0, buf);
    compare(expected, actual, "tBreak", 0, buf);
    compare(expected, actual, "numObs", 0, buf);
    compare(expected, actual, "category", 0, buf);
    compare(expected, actual, "changeProb", 1e-5, buf);
    //compareArrays(expected.coefs, actual.coefs, "coefs", tolerance, buf);
    compareArray(expected.rmse, actual.rmse, "rmse", tolerance, buf);
    compareArray(expected.magnitude, actual.magnitude, "magnitude", tolerance, buf);
    return (buf.length() == 0);
  }

  /** Compare the given field in the two object instances. */
  private void compare(Object expected, Object actual, String key, double tol, StringBuffer msg)
      throws Exception {
    Field f = expected.getClass().getField(key);
    double v1 = ((Number) f.get(expected)).doubleValue();
    double v2 = ((Number) f.get(actual)).doubleValue();
    if (Math.abs(v1 - v2) > tol) {
      msg.append(String.format(
          "Field %s mismatch. Expected '%s', found '%s'.\n",
          key, f.get(expected), f.get(actual)));
    }
  }

  /** Compare the two arrays to see that they match to within the given tolerance. */
  private void compareArrays(
      double[][] expected, double[][] actual, String key, double tol, StringBuffer msg) {
    for (int b = 0; b < expected.length; b++) {
      for (int p = 0; p < expected[0].length; p++) {
        double diff = Math.abs(expected[b][p] - actual[b][p]);
        if (diff > Math.max(tol, Math.abs(expected[b][p] * tol))) {
          msg.append(String.format(
              "%s Arrays differ at position [%d][%d] by %f (%f).  Expected %s, found %s\n",
              key, b, p, diff, tol, expected[b][p], actual[b][p]));
        }
      }
    }
  }

  /** Compare the two arrays to see that they match to within the given tolerance. */
  private void compareArray(
      double[] expected, double[] actual, String key, double tol, StringBuffer msg)  {
    for (int b = 0; b < expected.length; b++) {
      double diff = Math.abs(expected[b] - actual[b]);
      if (diff > Math.max(tol, Math.abs(expected[b] * tol))) {
        msg.append(String.format(
            "%s array differs at position [%d] by %f.\n", key, b, diff));
      }
    }
  }
}
