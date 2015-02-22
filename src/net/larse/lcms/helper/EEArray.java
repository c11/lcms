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

/**
 * This is used to mimic com.google.earthengine.api.array.EEArray
 */

//import java.util.List;

public class EEArray {
  private double[] array;

  public EEArray(double[] array) {
    this.array = array;
  }

  public static class Builder {
    private double[] array;

    public Builder(PixelType type, int[] lengths) {
      int size = 1;
      for (int i = 0; i < lengths.length; i++) {
        size *= lengths[i];
      }
      this.array = new double[size];
    }

    public EEArray build() {
      return new EEArray(array);
    }

    public void setDouble(double value, int offset) {
      array[offset] = value;
    }
  }

  public static EEArray.Builder builder(PixelType type, int... lengths) {
    return new Builder(type, lengths);
  }
}

