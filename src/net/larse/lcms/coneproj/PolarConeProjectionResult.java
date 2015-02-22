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
package net.larse.lcms.coneproj;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by yang on 2/17/15.
 *
 * This class represents the result from Cone Projection - Polar Cone.
 *
 * TODO: refactory this class with ConstraintConeProjectionResult.
 *
 * ConeProj.coneA()
 *
 */
public class PolarConeProjectionResult {
  /**
   * The dimension of the face of the constraint cone on which the projection lands.
   */
  public double df;

  /**
   * The projection of y on the constraint cone
   */
  public SimpleMatrix thetahat;

  /**
   * The number of iterations before the algorithm converges.
   */
  public int steps;

  public PolarConeProjectionResult(double df, SimpleMatrix thetahat, int steps) {
    this.df = df;
    this.thetahat = thetahat;
    this.steps = steps;
  }
}
