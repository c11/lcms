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
package net.larse.lcms.algorithms;

/**
 * Created by yang on 2/14/15.
 * 
 * This is part of LCMS project base learner implementation.
 *
 */

import com.google.common.primitives.Doubles;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.LUDecompositionImpl;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.RealVector;
import org.apache.commons.math.stat.descriptive.moment.Mean;
import org.apache.commons.math.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math.stat.regression.OLSMultipleLinearRegression;

import java.util.Arrays;
import java.util.List;

/**
 * Implements the EWMACD algorithm in the paper:
 * Brooks, E.B., R.H. Wynne, V.A. Thomas, C.E. Blinn, and J.W. Coulston. 2014.
 * On-the-fly Massively Multitemporal Change Detection
 * Using Statistical Quality Control Charts and Landsat Data,
 * IEEE Transaction on Geoscience and Remote Sensing, 52: 3316-3332
 *
 *
 * EWMACD is an algorithms that use Shewhart X-bar charts for changedetection.
 *
 *
 * @auther Zhiqiang Yang, 2/14/2015
 *
 * TODO:
 *  - consolidate the use of DoubleArrayList and RealVector
 *
 */
public class Ewmacd {

  public final static class EwmacdSolver {
    private final int DEFAULT_VALUE = -2222;
    //PARAMETER SECTION
    private final int harmonicCount;
    private final double xBarLimit1;
    private final double xBarLimit2;
    private final double vegetationThreshold;
    private final double lambda;
    private final double lambdasigs;
    private final boolean rounding;
    private final int persistence;
    //inclusive
    private final int trainingStartYear;
    //exclusive
    private final int trainingEndYear;

    public EwmacdSolver() {
      this.harmonicCount = 2;
      this.xBarLimit1 = 1.5;
      this.xBarLimit2 = 20.0;
      this.vegetationThreshold = 100;
      this.lambda = 0.3;
      this.lambdasigs = 3;
      this.rounding = true;
      this.persistence = 3;
      this.trainingStartYear = 2005;
      this.trainingEndYear = 2007;
    }

    public EwmacdSolver(int harmonicCount, double xBarLimit1, double xBarLimit2,
                        double vegetationThreshold, double lambda, double lambdasigs,
                        boolean rounding, int persistence,
                        int trainingStartYear, int trainingEndYear) {
      this.harmonicCount = harmonicCount;
      this.xBarLimit1 = xBarLimit1;
      this.xBarLimit2 = xBarLimit2;
      this.vegetationThreshold = vegetationThreshold;
      this.lambda = lambda;
      this.lambdasigs = lambdasigs;
      this.rounding = rounding;
      this.persistence = persistence;
      this.trainingStartYear = trainingStartYear;
      this.trainingEndYear = trainingEndYear;
    }

    /**
     * The main working horse of the algorithms
     *
     * In this implementation, it is assumed that nonvalid data has already been masked.
     * Need to confirm that when Google PixelFunction read values, the masked values would
     * not be returned here.
     *
     * @param x day of year for all the observations
     * @param y spectral values
     * @param years corresponding years
     * @return
     */
    //public List<Integer> getResult(double x, double y) {}
    public int[] getResult(double[] x, double[] y, double[] years) {

      //TODO: For Noel - using pixelfunction with (x,y) and return spectral vlaue and date information
      //In pixel funciton need to read in three set variables:
      // x: day of year for each image
      // y: spectral value for each image
      // years: image year for each image

      //TODO: do we need to handle masked value? not sure how pixelfunction behaves here.

      //EWMACD results
      int[] results = new int[x.length];

      //build a record of used indices
      int[] validObservations = new int[x.length];

      for (int i = 0; i < x.length; i++) {
        validObservations[i] = i;
        results[i] = DEFAULT_VALUE;
      }

      //find out the range of training data
      int trainingStart = 0;
      int trainingEnd = 0;
      for (int i = 0; i < years.length; i++) {
        if (years[i]==trainingStartYear) {
          trainingStart = i;
          break;
        }
      }
      for (int i = trainingStart; i < years.length; i++) {
        if (years[i] == trainingEndYear) {
          trainingEnd = i - 1;
          break;
        }
      }
      if (trainingEnd==0) {
        trainingEnd = years.length-1;
      }

      double[] trainingX = Arrays.copyOfRange(x, trainingStart, trainingEnd+1);
      double[] trainingY = Arrays.copyOfRange(y, trainingStart, trainingEnd+1);
      double[] betas = getTrainingHarmonic(trainingX, trainingY);

      // The following implements EWMA components
      //calculated fitted values for all the data points based on training coefficients
      RealMatrix all = constructHarmonicMatrix(x);
      RealMatrix fitted = all.multiply(new Array2DRowRealMatrix(betas));

      RealVector residuals = new Array2DRowRealMatrix(y).subtract(fitted).getColumnVector(0);
      int trainingSize = trainingEnd-trainingStart+1;
      RealVector trainingResiduals = residuals.getSubVector(trainingStart, trainingSize);

      //first estimate historical mean of residuals (should be near 0)
      double historicalStd = new StandardDeviation().evaluate(trainingResiduals.getData());

      //Original comments
      //Modifying SD estimates based on anomalous readings in the training data
      //We don't want to filter out the changes in the testing data, so xBarLimit2 is much larger!
      //The original implementation use only the first few years as training, do we need to handle if training
      //years are in the middle?
      double[] ucl0 = new double[x.length];
      for (int i = 0; i < x.length; i++) {
        if (years[i]<trainingEndYear && years[i] >= trainingStartYear) {
          ucl0[i] = xBarLimit1 * historicalStd;
        }
        else {
          ucl0[i] = xBarLimit2 * historicalStd;
        }
      }

      //Keeping only dates for which we have some vegetation and aren't anomalously far from 0 in the residuals
      DoubleArrayList filteredResiduals = new DoubleArrayList();
      IntArrayList usedObservations = new IntArrayList(); // this should be the final list used

      for (int i = 0; i < x.length; i++) {
        if (y[i] > vegetationThreshold && Math.abs(residuals.getEntry(i)) < ucl0[i]) {
          filteredResiduals.add(residuals.getEntry(i));
          usedObservations.add(validObservations[i]);
        }
      }

      //Do we have enough data to work with
      // I added this condition here. Original implementaiton checks this after ewma and control limite calculation,
      // which seems not efficient.
      if (filteredResiduals.size() <=3 ) {
        return adjustResult(results);
      }

      //updating historicalMean
      DoubleArrayList filteredTrainingResiduals = new DoubleArrayList();
      for (int i = 0; i < trainingResiduals.getDimension(); i++) {
        if (trainingY[i] > vegetationThreshold && Math.abs(trainingResiduals.getEntry(i)) < ucl0[i]) {
          filteredTrainingResiduals.add(trainingResiduals.getEntry(i));
        }
      }

      //do we have enough data to continue
      //TODO: original implementation only checks for standard deviation,
      // which can be calculated with just 2 numbers, should we make this a parameter like minimum observation
      // required?
      // original implementation return array with default NODATA value for all the years
      if (filteredTrainingResiduals.size()<2) {
        return adjustResult(results);
      }
      historicalStd = new StandardDeviation().evaluate(filteredTrainingResiduals.toDoubleArray());

      //Future EWMA output
      double[] ewma = new double[filteredResiduals.size()];
      ewma[0] = filteredResiduals.get(0);

      for (int i = 1; i < ewma.length; i++) {
        ewma[i] = ewma[i-1] * (1 - lambda) + lambda * filteredResiduals.get(i);
      }

      //EWMA upper control limit.  This is the threshold which dictates when the chart signals a disturbance.
      double[] upperControlLimit = new double[filteredResiduals.size()];
      for (int i = 0; i < upperControlLimit.length; i++) {
        upperControlLimit[i] = historicalStd * lambdasigs * Math.sqrt(lambda / (2 - lambda) * (1 - Math.pow((1 - lambda), 2 * (i + 1))));
      }

      //Integer value for EWMA output relative to control limit (rounded towards 0).
      //A value of +/-1 represents the weakest disturbance signal
      //coded values for the "present" subset of data
      int[] tmp2 = new int[ewma.length];
      for (int i = 0; i < tmp2.length; i++) {
        if (rounding) {
          tmp2[i] = (int)(Math.signum(ewma[i]) * Math.floor(Math.abs(ewma[i]/upperControlLimit[i])));
        }
        else {
          tmp2[i] = (int)(Math.round(ewma[i]));
        }
      }

      //Keeping only values for which a disturbance is sustained, using persistence as the threshold
      //NOTE: original implementation check this rules here, the check on tmp2.length should happen earlier to improve performance
      // it is redundant here, as I have moved it up right after filtering the data
      if (persistence > 1 && tmp2.length > 3) {
        //TODO: optimize the following section
        int[] tmp3 = new int[tmp2.length];
        int previousKeep = -1;
        for (int i = 0; i < tmp3.length; i++) {
          int low = 0;
          int high = 0;
          int score;
          while (i - low >= 0) {
            if (Math.signum(tmp2[i]) == Math.signum(tmp2[i - low])) {
              low++;
            } else {
              break;
            }
          }

          while (high + i < tmp3.length) {
            if (Math.signum(tmp2[i + high]) == Math.signum(tmp2[i])) {
              high++;
            } else {
              break;
            }
          }
          score = low + high - 1;

          //If sustained dates are long enough, keep; otherwise set to previous sustained state
          if (score >= persistence) {
            tmp3[i] = tmp2[i];
            previousKeep = i;
          }
          else {
            //NOTE: YANG - the following line would cause some minor difference with the original implementation.
            // where is original implmentation, it always assign the left most instance where sustained dates
            // pass the persistence test. I think the original implementation was wrong.
            // check for correctness
            tmp3[i] = previousKeep >= 0 ? tmp2[previousKeep] : 0;
          }
        }
        tmp2 = tmp3;
      }

      assert usedObservations.size() == tmp2.length;
      for (int i = 0; i < tmp2.length; i++) {
        results[usedObservations.getInt(i)] = tmp2[i];
      }

      return adjustResult(results);
    }

    /**
     * adjust EWMACD values for final result.
     * @param results
     * @return int[] filtered ewma values
     */
    private int[] adjustResult(int[] results) {
      //If the first date of myPixel was missing/filtered, then assign the EWMA output as 0 (no disturbance)
      if (results[0]==DEFAULT_VALUE) {
        results[0] = 0;
      }

      //Original comment: If we have EWMA information for the first date, then for each missing/filtered date
      // in the record, fill with the last known EWMA value
      // Note: with the above line, the following will always run
      for (int i = 1; i < results.length; i++) {
        if (results[i] == DEFAULT_VALUE) {
          results[i] = results[i-1];
        }
      }

      return results;
    }

    /**
     * derive harmonic coefficients using training data
     * @param trainingX
     * @param trainingY
     * @return
     */
    private double[] getTrainingHarmonic(double[] trainingX, double[] trainingY) {
      //construct training data
      Array2DRowRealMatrix matrix = constructHarmonicMatrix(trainingX);

      //make sure the design matrix has sufficient rank
      boolean det = isSingular(matrix);
      if (trainingX.length < (1 + 2 * harmonicCount) && isSingular(matrix)) {
        return null;
      }

      //first pass of harmonic function
      OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
      ols.newSampleData(trainingY, matrix.getSubMatrix(0, trainingX.length - 1, 1, matrix.getColumnDimension() - 1).getData());

      //check residual from first pass of harmonic funciton
      double[] residuals = ols.estimateResiduals();
      StandardDeviation sd = new StandardDeviation();
      double esd = sd.evaluate(residuals);

      //exclude observations with large residules: xBarLimit1 * sd
      int validCount = 0;
      DoubleArrayList samples2 = new DoubleArrayList();
      for (int i=0; i < residuals.length; i++) {
        if (Math.abs(residuals[i]) <= esd * xBarLimit1) {
          samples2.add(trainingY[i]);
          double rx = trainingX[i] * 2 * Math.PI / 365;
          samples2.add(Math.sin(rx));
          samples2.add(Math.cos(rx));
          samples2.add(Math.sin(2*rx));
          samples2.add(Math.cos(2*rx));

          validCount++;
        }
      }

      //refit the model using filtered training data
      ols.newSampleData(samples2.toDoubleArray(), validCount, 2 * harmonicCount);
      double[] params = ols.estimateRegressionParameters();

      return params;
    }


    protected Array2DRowRealMatrix constructHarmonicMatrix(double[] x) {
      Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(x.length, 2*harmonicCount+1);

      for (int i = 0; i < x.length; i++) {
        matrix.addToEntry(i, 0, 1);
        double rx = x[i] * 2 * Math.PI / 365;
        //now add the harmonic terms
        for (int h = 0; h < harmonicCount; h++) {
          matrix.addToEntry(i, h*2+1, Math.sin((h+1)*rx));
          matrix.addToEntry(i, h*2+2, Math.cos((h+1)*rx));
        }
      }

      return matrix;
    }

    protected boolean isSingular(Array2DRowRealMatrix matrix) {
      double singularThreshold = 0.001;
      RealMatrix prod = matrix.transpose().multiply(matrix);
      double determinant = new LUDecompositionImpl(prod).getDeterminant();

      return Math.abs(determinant) <= singularThreshold;
    }
  }
}
