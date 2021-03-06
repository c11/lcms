/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.larse.lcms.helper;

import java.text.DecimalFormat;
import java.util.List;

/**
 * Utility Math functions that are used by other classes.
 * 
 * @author Yasser Ganjisaffar (ganjisaffar at gmail dot com)
 *
 */
public class MathUtil {

    public static double getAvg(double[] arr) {
        double sum = 0;
        for (double item : arr) {
            sum += item;
        }
        return sum / arr.length;
    }

    public static double getAvg(float[] arr) {
        double sum = 0;
        for (double item : arr) {
            sum += item;
        }
        return sum / arr.length;
    }

    public static double getAvg(List<Double> arr) {
        double sum = 0;
        for (double item : arr) {
            sum += item;
        }
        return sum / arr.size();
    }

    public static double getStg(double[] arr) {
        return getStd(arr, getAvg(arr));
    }

    public static double getStg(List<Double> arr) {
        return getStd(arr, getAvg(arr));
    }

    public static double getStd(double[] arr, double avg) {
        double sum = 0;
        for (double item : arr) {
            sum += Math.pow(item - avg, 2);
        }
        return Math.sqrt(sum / arr.length);
    }

    public static double getStd(List<Double> arr, double avg) {
        double sum = 0;
        for (double item : arr) {
            sum += Math.pow(item - avg, 2);
        }
        return Math.sqrt(sum / arr.size());
    }

    public static double getDotProduct(float[] vector1, float[] vector2, int length) {
        double product = 0;
        for (int i = 0; i < length; i++) {
            product += vector1[i] * vector2[i];
        }
        return product;
    }

    public static double getDotProduct(double[] vector1, double[] vector2, int length) {
        double product = 0;
        for (int i = 0; i < length; i++) {
            product += vector1[i] * vector2[i];
        }
        return product;
    }

    public static double getDotProduct(double[] vector1, double[] vector2) {
        return getDotProduct(vector1, vector2, vector1.length);
    }

    // Divides the second vector from the first one (vector1[i] /= val)
    public static void divideInPlace(double[] vector, double val) {
        int length = vector.length;
        for (int i = 0; i < length; i++) {
            vector[i] /= val;
        }
    }

    public static double[][] allocateDoubleMatrix(int m, int n) {
        double[][] mat = new double[m][];
        for (int i = 0; i < m; i++) {
            mat[i] = new double[n];
        }
        return mat;
    }

    public static String getFormattedDouble(double val, int decimalPoints) {
        String format = "#.";
        for (int i = 0; i < decimalPoints; i++) {
            format += "#";
        }
        return new DecimalFormat(format).format(val);
    }
}
