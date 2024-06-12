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

import net.larse.lcms.helper.MathUtil;

/**
 * This class is a container for arrays and values that are computed during computation
 * of a lasso fit. It also contains the final weights of features.
 *
 * @author Yasser Ganjisaffar (ganjisaffar at gmail dot com)
 *     <p>Yang Z. modified to fit calculation needs for CCDC.
 */
public class LassoFit {
  // Number of lambda values
  public int numberOfLambdas;

  // Intercepts
  public double[] intercepts;

  // Compressed weights for each solution
  public double[][] compressedWeights;

  // Pointers to compressed weights
  public int[] indices;

  // Number of weights for each solution
  public int[] numberOfWeights;

  // Number of non-zero weights for each solution
  public int[] nonZeroWeights;

  // The value of lambdas for each solution
  public double[] lambdas;

  // R^2 value for each solution
  public double[] rsquared;

  // rmse for each solution
  public double[] rmses;
  public double[] adjustedRmses;

  // Total number of passes over data
  public int numberOfPasses;

  private int numFeatures;

  public LassoFit(int numberOfLambdas, int maxAllowedFeaturesAlongPath, int numFeatures) {
    intercepts = new double[numberOfLambdas];
    compressedWeights = new double[numberOfLambdas][maxAllowedFeaturesAlongPath];
    indices = new int[maxAllowedFeaturesAlongPath];
    numberOfWeights = new int[numberOfLambdas];
    lambdas = new double[numberOfLambdas];
    rsquared = new double[numberOfLambdas];
    rmses = new double[numberOfLambdas];
    adjustedRmses = new double[numberOfLambdas];
    nonZeroWeights = new int[numberOfLambdas];
    this.numFeatures = numFeatures;
  }

  public double[] getWeights(int lambdaIdx) {
    double[] weights = new double[numFeatures];
    for (int i = 0; i < numberOfWeights[lambdaIdx]; i++) {
      weights[indices[i]] = compressedWeights[lambdaIdx][i];
    }
    return weights;
  }

  /**
   * find the index corresponding to specified lambda
   *
   * <p>lambdas are stored in descending order.
   *
   * @param lambda
   * @return
   */
  public int getFitByLambda(double lambda) {
    int index = -1;

    // lambda is greater than the largest
    if (lambda >= lambdas[0]) {
      index = 0;
    } else if (lambda <= lambdas[lambdas.length - 1]) {
      index = lambdas.length - 1;
    } else {
      index = 0;
      double distance = Math.abs(lambda - lambdas[0]);
      for (int i = 1; i < lambdas.length; i++) {
        double cdist = Math.abs(lambda - lambdas[i]);
        if (cdist < distance) {
          index = i;
          distance = cdist;
        } else {
          break;
        }
      }
    }
    return index;
  }

  public double[] getBetas(int idx) {
    // intercept + number of features
    double[] result = new double[numFeatures + 1];
    result[0] = intercepts[idx];
    for (int i = 0; i < numberOfWeights[idx]; i++) {
      result[indices[i] + 1] = compressedWeights[idx][i];
    }
    return result;
  }

  public String toString() {
    StringBuilder sb = new StringBuilder();
    int numberOfSolutions = numberOfLambdas;
    sb.append("Compression R2 values:\n");
    for (int i = 0; i < numberOfSolutions; i++) {
      sb.append((i + 1));
      sb.append("\t");
      sb.append(nonZeroWeights[i]);
      sb.append("\t");
      sb.append(MathUtil.getFormattedDouble(rsquared[i], 4));
      sb.append("\t");
      sb.append(MathUtil.getFormattedDouble(lambdas[i], 5));
      sb.append("\n");
    }
    return sb.toString().trim();
  }
}
